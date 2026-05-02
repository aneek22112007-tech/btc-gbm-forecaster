"""
AlphaI × Polaris Challenge — BTC GBM Forecaster
================================================
Part A: 30-day backtest  |  Produces backtest_results.jsonl
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import requests, json, time
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

# ── 1. Data fetching ──────────────────────────────────────────────────────────
BINANCE_BASE = "https://data-api.binance.vision/api/v3/klines"

def fetch_btc_hourly(n_bars: int = 800, symbol: str = "BTCUSDT", interval: str = "1h") -> pd.DataFrame:
    """
    Fetch the last n_bars of BTCUSDT 1-hour candles from Binance public API.
    No API key needed.  Uses data-api.binance.vision to avoid Indian geo-block.
    Returns a DataFrame with columns: open_time, open, high, low, close, volume.
    """
    all_bars = []
    limit = 1000  # Binance max per request
    end_time = None  # start from now, walk backwards

    while len(all_bars) < n_bars:
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, n_bars - len(all_bars))}
        if end_time:
            params["endTime"] = end_time

        resp = requests.get(BINANCE_BASE, params=params, timeout=15)
        resp.raise_for_status()
        bars = resp.json()
        if not bars:
            break
        all_bars = bars + all_bars
        end_time = bars[0][0] - 1   # go further back
        if len(bars) < limit:
            break
        time.sleep(0.2)

    df = pd.DataFrame(all_bars, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","n_trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df[["open_time","open","high","low","close","volume"]]


# ── 2. GBM / Student-t model ──────────────────────────────────────────────────
def fit_and_predict(closes: np.ndarray, n_sim: int = 10_000,
                    conf: float = 0.95, lookback: int = 100,
                    vol_lookback: int = 20) -> tuple[float, float, float]:
    """
    Given an array of past closing prices (most-recent LAST):
      1. Compute log-returns.
      2. Estimate recent volatility using the last `vol_lookback` returns.
      3. Fit a Student-t distribution to returns.
      4. Simulate n_sim GBM paths one step forward.
      5. Return (low, high, current_price) for the `conf` interval.

    KEY: only data in `closes` is used — no future peeking possible.
    """
    if len(closes) < vol_lookback + 2:
        raise ValueError("Not enough data")

    log_ret = np.diff(np.log(closes))

    # Recent volatility (volatility clustering: use short window)
    recent_vol = np.std(log_ret[-vol_lookback:], ddof=1)

    # Fit Student-t to longer history for fat-tail ν estimate
    fit_data = log_ret[-min(lookback, len(log_ret)):]
    nu, mu_t, sigma_t = student_t.fit(fit_data, floc=0)  # fix loc=0

    # Use recent vol as scale, but Student-t ν for tail shape
    scale = recent_vol

    # Simulate one step
    S0 = closes[-1]
    sim_returns = student_t.rvs(df=nu, loc=mu_t, scale=scale, size=n_sim)
    sim_prices = S0 * np.exp(sim_returns)

    alpha = (1 - conf) / 2
    low = float(np.quantile(sim_prices, alpha))
    high = float(np.quantile(sim_prices, 1 - alpha))
    return low, high, float(S0)


# ── 3. Evaluation helpers ─────────────────────────────────────────────────────
def winkler_score(low: float, high: float, actual: float, alpha: float = 0.05) -> float:
    """
    Winkler interval score (lower = better).
    If actual is inside [low, high]: score = width
    If actual < low:  score = width + (2/alpha) * (low - actual)
    If actual > high: score = width + (2/alpha) * (actual - high)
    """
    width = high - low
    if actual < low:
        return width + (2 / alpha) * (low - actual)
    elif actual > high:
        return width + (2 / alpha) * (actual - high)
    else:
        return width


def evaluate(predictions: list[dict]) -> dict:
    """
    predictions: list of dicts with keys low, high, actual (all floats).
    Returns coverage_95, mean_width, mean_winkler_95.
    """
    hits, widths, winklers = [], [], []
    for p in predictions:
        low, high, actual = p["low"], p["high"], p["actual"]
        hits.append(1 if low <= actual <= high else 0)
        widths.append(high - low)
        winklers.append(winkler_score(low, high, actual))
    return {
        "coverage_95": float(np.mean(hits)),
        "mean_width": float(np.mean(widths)),
        "mean_winkler_95": float(np.mean(winklers)),
        "n_predictions": len(predictions),
    }


# ── 4. Part A: 30-day backtest ────────────────────────────────────────────────
def run_backtest(out_file: str = "backtest_results.jsonl",
                 n_bars: int = 800,
                 warmup: int = 120,   # bars needed before first prediction
                 vol_lookback: int = 20,
                 conf: float = 0.95) -> dict:
    """
    Fetch data, run walk-forward backtest, save JSONL, return metrics.
    We fetch 800 bars so after 720-bar backtest we still have warmup room.
    """
    print("Fetching BTC hourly data …")
    df = fetch_btc_hourly(n_bars)
    print(f"  Got {len(df)} bars. Range: {df.open_time.iloc[0]} → {df.open_time.iloc[-1]}")

    closes = df["close"].values
    timestamps = df["open_time"].tolist()

    predictions = []
    # Walk forward: predict bar i using only data up to bar i-1
    # Start from index `warmup` to have enough history
    start_idx = warmup
    end_idx = len(closes) - 1   # last bar is the target for second-to-last prediction

    print(f"Running backtest on bars {start_idx} → {end_idx-1} (predicting bar {start_idx+1} → {end_idx}) …")

    for i in range(start_idx, end_idx):
        # STRICTLY no peeking: use closes[0..i], predict closes[i+1]
        history = closes[:i+1]           # indices 0 … i  (NO bar i+1)
        actual  = closes[i+1]            # the future bar we predict

        try:
            low, high, s0 = fit_and_predict(
                history,
                vol_lookback=vol_lookback,
                conf=conf,
            )
        except Exception as e:
            print(f"  Warning: bar {i} failed ({e}), skipping")
            continue

        predictions.append({
            "bar_index": int(i),
            "prediction_time": timestamps[i].isoformat(),
            "target_time": timestamps[i+1].isoformat(),
            "current_price": float(s0),
            "low": low,
            "high": high,
            "actual": float(actual),
            "hit": bool(low <= actual <= high),
            "width": high - low,
            "winkler": winkler_score(low, high, float(actual)),
        })

        if (i - start_idx) % 100 == 0:
            print(f"  … {i - start_idx} / {end_idx - start_idx} done")

    # Save JSONL
    with open(out_file, "w") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")
    print(f"\nSaved {len(predictions)} predictions to '{out_file}'")

    metrics = evaluate(predictions)
    print("\n── Backtest Metrics ─────────────────────────────────────")
    print(f"  Coverage @95%  : {metrics['coverage_95']:.4f}  (target ≈ 0.95)")
    print(f"  Mean width     : ${metrics['mean_width']:,.2f}")
    print(f"  Mean Winkler   : ${metrics['mean_winkler_95']:,.2f}  (lower = better)")
    print(f"  N predictions  : {metrics['n_predictions']}")
    print("─────────────────────────────────────────────────────────")
    return metrics


# ── 5. Live prediction (used by dashboard) ────────────────────────────────────
def get_live_prediction(n_history: int = 500, conf: float = 0.95) -> dict:
    """
    Fetch last n_history bars and return a live 95% prediction for the next hour.
    """
    df = fetch_btc_hourly(n_history)
    closes = df["close"].values
    ts = df["open_time"].tolist()

    low, high, s0 = fit_and_predict(closes, conf=conf)
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "current_bar_time": ts[-1].isoformat(),
        "current_price": s0,
        "low_95": low,
        "high_95": high,
        "width": high - low,
        "df": df,   # full DataFrame for charting (not serialized)
    }


# ── 6. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    metrics = run_backtest()
    print("\nRun complete. Upload backtest_results.jsonl to your repo.")
