"""
AlphaI × Polaris Challenge — BTC Live Dashboard
================================================
Part B: Live prediction dashboard
Part C: Persistent prediction history (bonus)

Deploy: streamlit run dashboard.py
Host:   Streamlit Community Cloud (free) — connect your GitHub repo
"""

import streamlit as st
import requests, json, os, time
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from datetime import datetime, timezone
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC GBM Forecaster",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
BINANCE_BASE   = "https://data-api.binance.vision/api/v3/klines"
HISTORY_FILE   = "prediction_history.jsonl"   # Part C persistence
BACKTEST_FILE  = "backtest_results.jsonl"      # Pre-computed Part A results
CHART_BARS     = 50
N_HISTORY      = 500
CONF           = 0.95
VOL_LOOKBACK   = 20
LOOKBACK       = 100
N_SIM          = 10_000


# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)   # refresh at most every 60 s
def fetch_btc_hourly(n_bars: int = N_HISTORY) -> pd.DataFrame:
    all_bars, end_time = [], None
    while len(all_bars) < n_bars:
        params = {
            "symbol": "BTCUSDT", "interval": "1h",
            "limit": min(1000, n_bars - len(all_bars))
        }
        if end_time:
            params["endTime"] = end_time
        resp = requests.get(BINANCE_BASE, params=params, timeout=15)
        resp.raise_for_status()
        bars = resp.json()
        if not bars:
            break
        all_bars = bars + all_bars
        end_time = bars[0][0] - 1
        if len(bars) < 1000:
            break
    df = pd.DataFrame(all_bars, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","n_trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df.sort_values("open_time").reset_index(drop=True)


def fit_and_predict(closes: np.ndarray) -> tuple:
    log_ret     = np.diff(np.log(closes))
    recent_vol  = np.std(log_ret[-VOL_LOOKBACK:], ddof=1)
    fit_data    = log_ret[-min(LOOKBACK, len(log_ret)):]
    nu, mu_t, _ = student_t.fit(fit_data, floc=0)
    S0          = closes[-1]
    sim_ret     = student_t.rvs(df=nu, loc=mu_t, scale=recent_vol, size=N_SIM)
    sim_prices  = S0 * np.exp(sim_ret)
    alpha       = (1 - CONF) / 2
    return (
        float(np.quantile(sim_prices, alpha)),
        float(np.quantile(sim_prices, 1 - alpha)),
        float(S0),
    )


def winkler(low, high, actual, alpha=0.05):
    w = high - low
    if actual < low:   return w + (2/alpha)*(low - actual)
    elif actual > high: return w + (2/alpha)*(actual - high)
    return w


# ── Part C: persistence helpers ───────────────────────────────────────────────
def load_history() -> list:
    if not Path(HISTORY_FILE).exists():
        return []
    with open(HISTORY_FILE) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_history(record: dict):
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def fill_actuals(history: list, df: pd.DataFrame) -> list:
    """Back-fill actual prices into historical predictions when bar has closed."""
    if not history:
        return history
    price_map = {str(row.open_time): row.close for row in df.itertuples()}
    updated = []
    for rec in history:
        if rec.get("actual") is None:
            target = rec.get("target_bar_time")
            if target and target in price_map:
                rec = dict(rec)
                rec["actual"] = price_map[target]
                rec["hit"]    = rec["low_95"] <= rec["actual"] <= rec["high_95"]
                rec["winkler"]= winkler(rec["low_95"], rec["high_95"], rec["actual"])
        updated.append(rec)
    return updated


# ── Part A metrics from pre-computed backtest ─────────────────────────────────
@st.cache_data
def load_backtest_metrics():
    if not Path(BACKTEST_FILE).exists():
        return None
    rows = []
    with open(BACKTEST_FILE) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return None
    hits    = [r["hit"] for r in rows]
    widths  = [r["width"] for r in rows]
    winks   = [r["winkler"] for r in rows]
    return {
        "coverage_95":    round(float(np.mean(hits)), 4),
        "mean_width":     round(float(np.mean(widths)), 2),
        "mean_winkler_95": round(float(np.mean(winks)), 2),
        "n":              len(rows),
    }


# ── Chart builder ─────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, low: float, high: float, current: float) -> go.Figure:
    recent = df.tail(CHART_BARS).copy()
    next_time = recent["open_time"].iloc[-1] + pd.Timedelta(hours=1)

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=recent["open_time"],
        open=recent["open"], high=recent["high"],
        low=recent["low"],   close=recent["close"],
        name="BTCUSDT",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        showlegend=False,
    ))

    # Forecast ribbon
    fig.add_trace(go.Scatter(
        x=[recent["open_time"].iloc[-1], next_time, next_time, recent["open_time"].iloc[-1]],
        y=[current, high, low, current],
        fill="toself",
        fillcolor="rgba(99,179,237,0.18)",
        line=dict(color="rgba(99,179,237,0)"),
        name="95% range",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Upper / lower bounds
    fig.add_trace(go.Scatter(
        x=[next_time], y=[high],
        mode="markers+text",
        marker=dict(color="#63b3ed", size=10, symbol="line-ew"),
        text=[f"${high:,.0f}"], textposition="top right",
        name=f"Upper 95%: ${high:,.0f}",
    ))
    fig.add_trace(go.Scatter(
        x=[next_time], y=[low],
        mode="markers+text",
        marker=dict(color="#63b3ed", size=10, symbol="line-ew"),
        text=[f"${low:,.0f}"], textposition="bottom right",
        name=f"Lower 95%: ${low:,.0f}",
    ))

    fig.update_layout(
        title=f"BTCUSDT — Last {CHART_BARS} bars + next-hour 95% range",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        legend=dict(bgcolor="#1a1d24", bordercolor="#333"),
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(gridcolor="#1e2530"),
        yaxis=dict(gridcolor="#1e2530", tickprefix="$", tickformat=",.0f"),
    )
    return fig


def build_history_chart(history: list) -> go.Figure:
    rows = [r for r in history if r.get("actual") is not None]
    if len(rows) < 2:
        return None
    df_h = pd.DataFrame(rows)
    df_h["prediction_time"] = pd.to_datetime(df_h["prediction_time"])
    df_h = df_h.sort_values("prediction_time").tail(96)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_h["prediction_time"], y=df_h["actual"],
        mode="lines", name="Actual BTC", line=dict(color="#f6c90e", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=list(df_h["prediction_time"]) + list(df_h["prediction_time"])[::-1],
        y=list(df_h["high_95"]) + list(df_h["low_95"])[::-1],
        fill="toself", fillcolor="rgba(99,179,237,0.2)",
        line=dict(color="rgba(0,0,0,0)"), name="95% band",
    ))
    hits = df_h[df_h["hit"] == True]
    misses = df_h[df_h["hit"] == False]
    fig.add_trace(go.Scatter(
        x=hits["prediction_time"], y=hits["actual"],
        mode="markers", marker=dict(color="#26a69a", size=7, symbol="circle"),
        name="✓ Hit",
    ))
    fig.add_trace(go.Scatter(
        x=misses["prediction_time"], y=misses["actual"],
        mode="markers", marker=dict(color="#ef5350", size=9, symbol="x"),
        name="✗ Miss",
    ))
    fig.update_layout(
        title="Live prediction history (actuals filled in as bars close)",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        height=360, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(gridcolor="#1e2530"),
        yaxis=dict(gridcolor="#1e2530", tickprefix="$", tickformat=",.0f"),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    auto_refresh = st.checkbox("Auto-refresh every 60 s", value=False)
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("**Model parameters**")
    st.write(f"• Confidence: {int(CONF*100)}%")
    st.write(f"• Vol window: {VOL_LOOKBACK} bars")
    st.write(f"• Simulations: {N_SIM:,}")
    st.write(f"• History fed: {N_HISTORY} bars")
    st.markdown("---")
    st.caption("AlphaI × Polaris Challenge\nGBM + Student-t forecaster")


# ── Main render ───────────────────────────────────────────────────────────────
st.title("₿ BTC/USDT — GBM 95% Interval Forecaster")
st.caption(f"Refreshed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# ── Fetch data & predict ──────────────────────────────────────────────────────
with st.spinner("Fetching latest BTC data from Binance …"):
    try:
        df = fetch_btc_hourly()
        closes = df["close"].values
        low, high, current = fit_and_predict(closes)
        error_msg = None
    except Exception as e:
        error_msg = str(e)

if error_msg:
    st.error(f"Data fetch failed: {error_msg}")
    st.stop()

next_bar_time = df["open_time"].iloc[-1] + pd.Timedelta(hours=1)

# Part C: save this prediction
prediction_record = {
    "prediction_time":  datetime.now(timezone.utc).isoformat(),
    "current_bar_time": df["open_time"].iloc[-1].isoformat(),
    "target_bar_time":  next_bar_time.isoformat(),
    "current_price":    current,
    "low_95":           low,
    "high_95":          high,
    "width":            high - low,
    "actual":           None,
    "hit":              None,
    "winkler":          None,
}
append_history(prediction_record)

# Fill actuals in history
history = load_history()
history = fill_actuals(history, df)
# Re-save with filled actuals
with open(HISTORY_FILE, "w") as f:
    for rec in history:
        f.write(json.dumps(rec) + "\n")

# ── Part A backtest metrics (headline row) ────────────────────────────────────
bm = load_backtest_metrics()

st.markdown("### 📊 Backtest Metrics (30-day walk-forward)")
if bm:
    c1, c2, c3, c4 = st.columns(4)
    delta_cov = round(bm["coverage_95"] - 0.95, 4)
    c1.metric("Coverage @95%", f"{bm['coverage_95']:.4f}",
              delta=f"{delta_cov:+.4f} vs target",
              delta_color="normal")
    c2.metric("Mean Width", f"${bm['mean_width']:,.0f}")
    c3.metric("Mean Winkler ↓", f"${bm['mean_winkler_95']:,.0f}",
              help="Lower is better")
    c4.metric("Predictions", f"{bm['n']:,}")
else:
    st.info("Run `python btc_gbm_backtest.py` to generate `backtest_results.jsonl`, "
            "then re-deploy. Backtest metrics will appear here.")

st.divider()

# ── Live prediction ────────────────────────────────────────────────────────────
st.markdown("### 🔮 Next-Hour Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Current BTC Price", f"${current:,.2f}")
col2.metric("95% Range Lower", f"${low:,.2f}", delta=f"{(low/current - 1)*100:+.2f}%")
col3.metric("95% Range Upper", f"${high:,.2f}", delta=f"{(high/current - 1)*100:+.2f}%")

w_pct = (high - low) / current * 100
st.caption(f"Range width: **${high-low:,.0f}** ({w_pct:.2f}% of current price) · "
           f"Target bar closes at **{next_bar_time.strftime('%Y-%m-%d %H:%M UTC')}**")

# ── Main chart ────────────────────────────────────────────────────────────────
fig = build_chart(df, low, high, current)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Part C: History chart ──────────────────────────────────────────────────────
resolved = [r for r in history if r.get("actual") is not None]
st.markdown(f"### 📈 Prediction History — {len(resolved)} resolved, {len(history)} total")

if resolved:
    hist_hits   = [r for r in resolved if r.get("hit")]
    live_cov    = len(hist_hits) / len(resolved)
    live_winkl  = float(np.mean([r["winkler"] for r in resolved]))
    hc1, hc2, hc3 = st.columns(3)
    hc1.metric("Live Coverage", f"{live_cov:.4f}")
    hc2.metric("Live Winkler ↓", f"${live_winkl:,.0f}")
    hc3.metric("Resolved Preds", len(resolved))

    fig_h = build_history_chart(history)
    if fig_h:
        st.plotly_chart(fig_h, use_container_width=True)
else:
    st.info("Visit again after a few hours — predictions will accumulate here as bars close.")

# ── Raw history table (collapsible) ───────────────────────────────────────────
with st.expander("🗂️ Raw prediction log"):
    if history:
        df_hist = pd.DataFrame(history[::-1]).drop(columns=["actual"], errors="ignore")
        st.dataframe(df_hist, use_container_width=True, height=300)
    else:
        st.write("No history yet.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
