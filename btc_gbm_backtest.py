"""
AlphaI × Polaris Challenge — BTC GBM Backtest  (IMPROVED v2)
=============================================================
Run this locally or in Colab to regenerate backtest_results.jsonl.

Improvements over starter:
  • EWMA volatility (λ=0.94 RiskMetrics) — better volatility clustering
  • Adaptive nu via MLE fit on sliding window, clipped to [2.5, 6]
  • Itô drift correction (mu -= 0.5*sigma²) for unbiased GBM
  • 15k simulations for smoother quantile estimates
  • Detailed per-bar record including vol, nu, regime
"""

import requests, json, time, sys
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
BINANCE_BASE  = "https://data-api.binance.vision/api/v3/klines"
N_SIM         = 15_000
CONF          = 0.95
VOL_LOOKBACK  = 24        # EWMA window (effective half-life)
LOOKBACK      = 120       # nu MLE window
EWMA_LAMBDA   = 0.94      # RiskMetrics decay factor
WARMUP        = 150       # bars before first prediction
TARGET_BARS   = 720       # ~30 days of 1h bars

# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_btc_hourly(n_bars: int = TARGET_BARS + WARMUP + 50) -> pd.DataFrame:
    print(f"Fetching {n_bars} BTCUSDT 1h bars from Binance …")
    all_bars, end_time = [], None
    while len(all_bars) < n_bars:
        params = {"symbol": "BTCUSDT", "interval": "1h",
                  "limit": min(1000, n_bars - len(all_bars))}
        if end_time:
            params["endTime"] = end_time
        resp = requests.get(BINANCE_BASE, params=params, timeout=15)
        resp.raise_for_status()
        bars = resp.json()
        if not bars: break
        all_bars = bars + all_bars
        end_time = bars[0][0] - 1
        if len(bars) < 1000: break
        time.sleep(0.2)
    df = pd.DataFrame(all_bars, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","n_trades","taker_buy_base","taker_buy_quote","ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"Fetched {len(df)} bars: {df.open_time.iloc[0]}  →  {df.open_time.iloc[-1]}")
    return df

# ── Improved model ────────────────────────────────────────────────────────────

def ewma_vol(log_returns: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    """EWMA (RiskMetrics) volatility. Much more responsive than rolling std."""
    if len(log_returns) < 2:
        return float(np.std(log_returns, ddof=1)) if len(log_returns) > 0 else 1e-4
    var = float(np.var(log_returns[:min(10, len(log_returns))], ddof=1))
    for r in log_returns:
        var = lam * var + (1 - lam) * r**2
    return float(np.sqrt(max(var, 1e-12)))


def fit_nu_mle(log_returns: np.ndarray) -> float:
    """MLE for Student-t ν, clipped to empirical BTC range [2.5, 6]."""
    if len(log_returns) < 20:
        return 4.0
    scale = np.std(log_returns, ddof=1)
    if scale < 1e-10:
        return 4.0
    try:
        nu, _, _ = student_t.fit(log_returns, floc=0, fscale=scale)
        return float(np.clip(nu, 2.5, 6.0))
    except Exception:
        return 4.0


def detect_regime(log_returns: np.ndarray,
                  short_w: int = 12, long_w: int = 72) -> str:
    if len(log_returns) < long_w:
        return "neutral"
    v_short = np.std(log_returns[-short_w:], ddof=1)
    v_long  = np.std(log_returns[-long_w:],  ddof=1)
    ratio = v_short / (v_long + 1e-10)
    if ratio > 1.25: return "volatile"
    if ratio < 0.80: return "calm"
    return "neutral"


def fit_and_predict(closes: np.ndarray,
                    n_sim: int = N_SIM, conf: float = CONF) -> dict:
    """
    Improved prediction. Returns dict with low, high, current, vol, nu, regime.

    STRICT NO-PEEK: only `closes` is passed (caller ensures no future data).
    """
    log_ret = np.diff(np.log(closes))

    # EWMA volatility on recent bars
    vol_h   = ewma_vol(log_ret[-max(VOL_LOOKBACK * 2, 50):])

    # Adaptive nu via MLE
    fit_data = log_ret[-min(LOOKBACK, len(log_ret)):]
    nu = fit_nu_mle(fit_data)

    # Itô drift correction
    mu = float(np.mean(fit_data)) - 0.5 * vol_h**2

    # Regime
    regime = detect_regime(log_ret)

    S0 = float(closes[-1])
    sim_ret    = student_t.rvs(df=nu, loc=mu, scale=vol_h, size=n_sim)
    sim_prices = S0 * np.exp(sim_ret)

    alpha = (1 - conf) / 2
    low  = float(np.quantile(sim_prices, alpha))
    high = float(np.quantile(sim_prices, 1 - alpha))

    return dict(low=low, high=high, current=S0, vol_h=vol_h, nu=nu, regime=regime)

# ── Scoring ───────────────────────────────────────────────────────────────────

def winkler_score(low, high, actual, alpha=0.05):
    width = high - low
    if   actual < low:  return width + (2/alpha)*(low - actual)
    elif actual > high: return width + (2/alpha)*(actual - high)
    return width


def evaluate(predictions: list) -> dict:
    hits, widths, winklers = [], [], []
    for p in predictions:
        low, high, actual = p["low"], p["high"], p["actual"]
        hits.append(1 if low <= actual <= high else 0)
        widths.append(high - low)
        winklers.append(winkler_score(low, high, actual))
    return {
        "coverage_95":     float(np.mean(hits)),
        "mean_width":      float(np.mean(widths)),
        "mean_winkler_95": float(np.mean(winklers)),
        "n_predictions":   len(predictions),
    }

# ── Walk-forward backtest ─────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> list:
    closes     = df["close"].values
    timestamps = df["open_time"].tolist()
    predictions = []

    total = len(closes) - WARMUP - 1
    print(f"\nRunning {total} walk-forward predictions (WARMUP={WARMUP}) …")

    for i in range(WARMUP, len(closes) - 1):
        # ── STRICT NO-PEEK: only indices 0..i, actual = i+1 ──────────────
        history = closes[:i+1]    # does NOT include closes[i+1]
        actual  = float(closes[i+1])

        try:
            pred = fit_and_predict(history)
        except Exception as e:
            print(f"  [WARN] bar {i}: {e}")
            continue

        low, high = pred["low"], pred["high"]
        w = winkler_score(low, high, actual)

        predictions.append({
            "bar_index":       int(i),
            "prediction_time": timestamps[i].isoformat(),
            "target_time":     timestamps[i+1].isoformat(),
            "current_price":   pred["current"],
            "low":             low,
            "high":            high,
            "actual":          actual,
            "hit":             bool(low <= actual <= high),
            "width":           high - low,
            "winkler":         w,
            "vol_h":           pred["vol_h"],
            "nu":              pred["nu"],
            "regime":          pred["regime"],
        })

        done = i - WARMUP
        if done % 100 == 0:
            running_cov = np.mean([p["hit"] for p in predictions])
            print(f"  {done}/{total} … running coverage={running_cov:.4f}")

    return predictions

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = fetch_btc_hourly()

    predictions = run_backtest(df)

    metrics = evaluate(predictions)
    print("\n══ Backtest Metrics (IMPROVED v2) ══════════════════════════════")
    print(f"  Coverage @95%  : {metrics['coverage_95']:.4f}  (ideal ≈ 0.95)")
    print(f"  Mean width     : ${metrics['mean_width']:>12,.2f}")
    print(f"  Mean Winkler ↓ : ${metrics['mean_winkler_95']:>12,.2f}  (lower = better)")
    print(f"  N predictions  : {metrics['n_predictions']}")
    print("════════════════════════════════════════════════════════════════")

    # Per-regime breakdown
    for regime in ["calm", "neutral", "volatile"]:
        subset = [p for p in predictions if p.get("regime") == regime]
        if subset:
            cov = np.mean([p["hit"] for p in subset])
            wkl = np.mean([p["winkler"] for p in subset])
            print(f"  Regime '{regime}': n={len(subset)}, cov={cov:.4f}, winkler={wkl:,.0f}")

    out_file = "backtest_results.jsonl"
    with open(out_file, "w") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")

    print(f"\nSaved {len(predictions)} predictions to '{out_file}'")
    print("Commit this file to your GitHub repo — dashboard reads it for headline metrics.")