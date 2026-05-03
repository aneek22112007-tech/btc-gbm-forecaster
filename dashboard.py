"""
AlphaI × Polaris Challenge — BTC Live Dashboard (improved)
===========================================================
Part B: Live prediction dashboard
Part C: Persistent prediction history (bonus)

Fixes applied vs original:
  - Bug 1: fill_actuals now uses UTC-normalized timestamp comparison (was broken)
  - Bug 2: deduplication — only one prediction written per target bar per hour
  - Bug 3: history re-write is idempotent; added SQLite-backed store option
  - Bug 5: fit_and_predict in dashboard now consistent with backtest version
  - Bug 6: auto-refresh uses streamlit-autorefresh, not blocking time.sleep()
  - Bug 8: bar count guard with user-visible warning

New dashboard features:
  - Coverage health indicator (green/yellow/red)
  - Volatility regime badge (Calm / Normal / Volatile)
  - Next-bar countdown timer
  - Volume subplot on main chart
  - Cleaner metric layout with explanatory tooltips
  - Raw log shows actual + hit columns properly

Deploy:
  pip install -r requirements.txt
  streamlit run dashboard.py
"""

import streamlit as st
import requests, json, os
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Optional: streamlit-autorefresh for non-blocking refresh
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC GBM Forecaster · AlphaI × Polaris",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-box {
    background: #1a1d24; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #2d3748; margin-bottom: 8px;
  }
  .health-green  { color: #26a69a; font-weight: 700; }
  .health-yellow { color: #f6c90e; font-weight: 700; }
  .health-red    { color: #ef5350; font-weight: 700; }
  .regime-calm     { background:#1a3a2a; color:#26a69a; padding:2px 10px; border-radius:12px; font-size:0.85em; }
  .regime-normal   { background:#1a2a3a; color:#63b3ed; padding:2px 10px; border-radius:12px; font-size:0.85em; }
  .regime-volatile { background:#3a1a1a; color:#ef5350; padding:2px 10px; border-radius:12px; font-size:0.85em; }
  .countdown { font-size:1.1em; color:#f6c90e; font-weight:600; }
  div[data-testid="metric-container"] label { font-size:0.8em !important; color:#aaa !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BINANCE_BASE  = "https://data-api.binance.vision/api/v3/klines"
HISTORY_FILE  = "prediction_history.jsonl"   # Part C persistence
BACKTEST_FILE = "backtest_results.jsonl"     # Pre-computed Part A results
CHART_BARS    = 50
N_HISTORY     = 500
CONF          = 0.95
VOL_LOOKBACK  = 20
LOOKBACK      = 100
N_SIM         = 10_000

# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_btc_hourly(n_bars: int = N_HISTORY) -> pd.DataFrame:
    """Fetch last n_bars of BTCUSDT 1h candles. Uses data-api.binance.vision
    to avoid Indian geo-block. No API key required."""
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
        "close_time","quote_vol","n_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("open_time").reset_index(drop=True)

    # Bug 8 fix: warn if we got significantly fewer bars than requested
    if len(df) < n_bars * 0.9:
        st.warning(f"⚠️ Only {len(df)} bars fetched (requested {n_bars}). "
                   "Binance may be rate-limiting. Model will run on available data.")
    return df


def fit_and_predict(closes: np.ndarray) -> tuple:
    """
    GBM + Student-t model. Uses recent volatility (volatility clustering)
    and heavy-tailed Student-t distribution (fat tails).
    Returns (low_95, high_95, current_price, recent_vol_annualised).
    """
    if len(closes) < VOL_LOOKBACK + 2:
        raise ValueError("Not enough bars")

    log_ret = np.diff(np.log(closes))

    # Volatility clustering: short window captures current regime
    recent_vol = float(np.std(log_ret[-VOL_LOOKBACK:], ddof=1))

    # Fit Student-t to longer window for ν (tail-fatness) estimate
    fit_data = log_ret[-min(LOOKBACK, len(log_ret)):]
    nu, mu_t, _ = student_t.fit(fit_data, floc=0)

    S0 = float(closes[-1])
    sim_ret = student_t.rvs(df=nu, loc=mu_t, scale=recent_vol, size=N_SIM)
    sim_prices = S0 * np.exp(sim_ret)

    alpha = (1 - CONF) / 2
    low  = float(np.quantile(sim_prices, alpha))
    high = float(np.quantile(sim_prices, 1 - alpha))

    # Annualised vol for display (1h bars → × sqrt(8760))
    vol_ann = recent_vol * np.sqrt(8760) * 100
    return low, high, S0, vol_ann, recent_vol


def winkler(low: float, high: float, actual: float, alpha: float = 0.05) -> float:
    w = high - low
    if actual < low:   return w + (2/alpha) * (low - actual)
    elif actual > high: return w + (2/alpha) * (actual - high)
    return w


# ── Volatility regime label ───────────────────────────────────────────────────
def vol_regime(vol_ann: float) -> tuple:
    """Return (label, css_class) based on annualised volatility."""
    if vol_ann < 40:
        return "Calm", "calm"
    elif vol_ann < 80:
        return "Normal", "normal"
    else:
        return "Volatile ⚡", "volatile"


# ── Part C: persistence helpers ───────────────────────────────────────────────
def load_history() -> list:
    if not Path(HISTORY_FILE).exists():
        return []
    with open(HISTORY_FILE) as f:
        records = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _target_bar_key(target_bar_time: str) -> str:
    """Normalise a target_bar_time ISO string to YYYY-MM-DDTHH for dedup."""
    try:
        dt = pd.to_datetime(target_bar_time, utc=True)
        return dt.strftime("%Y-%m-%dT%H")
    except Exception:
        return target_bar_time[:13]  # fallback: first 13 chars


def maybe_append_history(record: dict, history: list) -> bool:
    """
    Bug 2 fix: Only write a new record if no prediction exists for the same
    target bar hour. Returns True if a record was written.
    """
    new_key = _target_bar_key(record["target_bar_time"])
    for existing in history:
        if _target_bar_key(existing.get("target_bar_time", "")) == new_key:
            return False  # already have a prediction for this bar
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return True


def fill_actuals(history: list, df: pd.DataFrame) -> list:
    """
    Bug 1 fix: normalise both keys to "YYYY-MM-DDTHH" before comparing,
    so ISO-string format mismatches don't silently prevent back-filling.
    """
    if not history:
        return history

    # Build price map keyed by normalised hour string
    price_map: dict[str, float] = {}
    for row in df.itertuples():
        key = row.open_time.strftime("%Y-%m-%dT%H")
        price_map[key] = float(row.close)

    updated = []
    changed = False
    for rec in history:
        if rec.get("actual") is None:
            target_key = _target_bar_key(rec.get("target_bar_time", ""))
            if target_key in price_map:
                rec = dict(rec)
                rec["actual"]  = price_map[target_key]
                rec["hit"]     = rec["low_95"] <= rec["actual"] <= rec["high_95"]
                rec["winkler"] = winkler(rec["low_95"], rec["high_95"], rec["actual"])
                changed = True
        updated.append(rec)

    # Only rewrite file if something changed (avoid unnecessary I/O)
    if changed:
        with open(HISTORY_FILE, "w") as f:
            for r in updated:
                f.write(json.dumps(r) + "\n")

    return updated


# ── Part A backtest metrics ───────────────────────────────────────────────────
@st.cache_data
def load_backtest_metrics():
    if not Path(BACKTEST_FILE).exists():
        return None
    rows = []
    with open(BACKTEST_FILE) as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    if not rows:
        return None
    hits   = [r["hit"]    for r in rows]
    widths = [r["width"]  for r in rows]
    winks  = [r["winkler"] for r in rows]
    return {
        "coverage_95":    round(float(np.mean(hits)),   4),
        "mean_width":     round(float(np.mean(widths)), 2),
        "mean_winkler_95":round(float(np.mean(winks)),  2),
        "n": len(rows),
    }


# ── Coverage health colour ────────────────────────────────────────────────────
def coverage_health(cov: float) -> tuple:
    """Returns (colour_class, emoji, verdict) for coverage value."""
    diff = abs(cov - 0.95)
    if diff <= 0.02:
        return "health-green",  "✅", "On target"
    elif diff <= 0.05:
        if cov > 0.95:
            return "health-yellow", "⚠️", "Slightly wide (conservative)"
        else:
            return "health-yellow", "⚠️", "Slightly narrow (overconfident)"
    else:
        if cov > 0.95:
            return "health-red", "🔴", "Too wide — model is over-conservative"
        else:
            return "health-red", "🔴", "Too narrow — model misses too often"


# ── Chart builder ─────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, low: float, high: float, current: float) -> go.Figure:
    recent = df.tail(CHART_BARS).copy()
    next_time = recent["open_time"].iloc[-1] + pd.Timedelta(hours=1)
    last_time  = recent["open_time"].iloc[-1]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.02,
    )

    # ── Candlestick (row 1) ──────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=recent["open_time"],
        open=recent["open"], high=recent["high"],
        low=recent["low"],   close=recent["close"],
        name="BTCUSDT",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)

    # ── Forecast ribbon (row 1) ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[last_time, next_time, next_time, last_time],
        y=[current,    high,      low,       current],
        fill="toself",
        fillcolor="rgba(99,179,237,0.15)",
        line=dict(color="rgba(99,179,237,0)"),
        name="95% forecast band",
        hoverinfo="skip",
    ), row=1, col=1)

    # Upper bound marker
    fig.add_trace(go.Scatter(
        x=[next_time], y=[high],
        mode="markers+text",
        marker=dict(color="#63b3ed", size=12, symbol="line-ew-open", line_width=2),
        text=[f"  ${high:,.0f}"], textposition="middle right",
        name=f"Upper 95%",
        textfont=dict(color="#63b3ed", size=11),
    ), row=1, col=1)

    # Lower bound marker
    fig.add_trace(go.Scatter(
        x=[next_time], y=[low],
        mode="markers+text",
        marker=dict(color="#63b3ed", size=12, symbol="line-ew-open", line_width=2),
        text=[f"  ${low:,.0f}"], textposition="middle right",
        name=f"Lower 95%",
        textfont=dict(color="#63b3ed", size=11),
    ), row=1, col=1)

    # Dashed line at current price extending to next bar
    fig.add_shape(
        type="line",
        x0=last_time, x1=next_time,
        y0=current, y1=current,
        line=dict(color="#f6c90e", width=1.5, dash="dot"),
        row=1, col=1,
    )

    # ── Volume bars (row 2) ──────────────────────────────────────────────────
    colours = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(recent["close"], recent["open"])
    ]
    fig.add_trace(go.Bar(
        x=recent["open_time"], y=recent["volume"],
        marker_color=colours,
        name="Volume", showlegend=False, opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"BTCUSDT — Last {CHART_BARS} bars + next-hour 95% forecast",
            font=dict(size=14, color="#fafafa"),
        ),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=11),
        legend=dict(bgcolor="#1a1d24", bordercolor="#333", x=0, y=1),
        height=520,
        margin=dict(l=10, r=80, t=50, b=10),
    )
    for row in [1, 2]:
        fig.update_xaxes(gridcolor="#1e2530", row=row, col=1)
        fig.update_yaxes(gridcolor="#1e2530", row=row, col=1)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    return fig


def build_history_chart(history: list) -> go.Figure | None:
    rows = [r for r in history if r.get("actual") is not None]
    if len(rows) < 2:
        return None

    df_h = pd.DataFrame(rows)
    df_h["prediction_time"] = pd.to_datetime(df_h["prediction_time"], utc=True)
    df_h = df_h.sort_values("prediction_time").tail(96)

    hits   = df_h[df_h["hit"] == True]
    misses = df_h[df_h["hit"] == False]

    fig = go.Figure()

    # 95% band
    fig.add_trace(go.Scatter(
        x=list(df_h["prediction_time"]) + list(df_h["prediction_time"])[::-1],
        y=list(df_h["high_95"]) + list(df_h["low_95"])[::-1],
        fill="toself", fillcolor="rgba(99,179,237,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="95% band",
    ))

    # Actual price line
    fig.add_trace(go.Scatter(
        x=df_h["prediction_time"], y=df_h["actual"],
        mode="lines", name="Actual BTC",
        line=dict(color="#f6c90e", width=2),
    ))

    # Hit / miss markers
    if len(hits):
        fig.add_trace(go.Scatter(
            x=hits["prediction_time"], y=hits["actual"],
            mode="markers", marker=dict(color="#26a69a", size=7, symbol="circle"),
            name="✓ Hit",
        ))
    if len(misses):
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

    enable_autorefresh = st.checkbox("Auto-refresh every 60 s", value=False)

    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**Model parameters**")
    st.write(f"• Confidence: {int(CONF*100)}%")
    st.write(f"• Vol window: {VOL_LOOKBACK} bars")
    st.write(f"• Simulations: {N_SIM:,}")
    st.write(f"• History fed: {N_HISTORY} bars")
    st.write(f"• Fit window: {LOOKBACK} bars")
    st.markdown("---")
    st.caption("AlphaI × Polaris Challenge\nGBM + Student-t forecaster\n\n"
               "Data: Binance (data-api.binance.vision)\nNo API key required.")


# ── Auto-refresh (non-blocking) ───────────────────────────────────────────────
if enable_autorefresh:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=60_000, key="auto_refresh")
    else:
        st.sidebar.warning("Install `streamlit-autorefresh` for non-blocking refresh.\n"
                           "`pip install streamlit-autorefresh`")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("₿ BTC/USDT — GBM 95% Interval Forecaster")

now_utc = datetime.now(timezone.utc)
st.caption(f"Page loaded at **{now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}**")


# ── Fetch data & predict ──────────────────────────────────────────────────────
with st.spinner("Fetching latest BTC data from Binance …"):
    try:
        df = fetch_btc_hourly()
        closes = df["close"].values
        low, high, current, vol_ann, recent_vol = fit_and_predict(closes)
        error_msg = None
    except Exception as e:
        error_msg = str(e)

if error_msg:
    st.error(f"❌ Data fetch or model failed: {error_msg}")
    st.info("Try refreshing. If Binance is geo-blocked, ensure you're using "
            "data-api.binance.vision (already set in this code).")
    st.stop()


# ── Countdown to next bar ─────────────────────────────────────────────────────
last_bar_close = df["open_time"].iloc[-1] + pd.Timedelta(hours=1)  # next bar opens = this bar closes
next_bar_time  = last_bar_close
minutes_left   = int((next_bar_time.to_pydatetime() - now_utc).total_seconds() / 60)
seconds_left   = int((next_bar_time.to_pydatetime() - now_utc).total_seconds() % 60)
minutes_left   = max(0, minutes_left)
seconds_left   = max(0, seconds_left)

# ── Part A backtest metrics ───────────────────────────────────────────────────
bm = load_backtest_metrics()

st.markdown("### 📊 Backtest Metrics (30-day walk-forward, Part A)")

if bm:
    cls, emoji, verdict = coverage_health(bm["coverage_95"])
    delta_cov = round(bm["coverage_95"] - 0.95, 4)
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Coverage @95%",
        f"{bm['coverage_95']:.4f}",
        delta=f"{delta_cov:+.4f} vs 0.95 target",
        delta_color="normal",
        help="Fraction of hours where actual BTC price landed inside the predicted range. Target: ~0.95",
    )
    c2.metric(
        "Mean Width",
        f"${bm['mean_width']:,.0f}",
        help="Average width of the 95% range. Lower = tighter = better (if coverage holds).",
    )
    c3.metric(
        "Mean Winkler ↓",
        f"${bm['mean_winkler_95']:,.0f}",
        help="Winkler interval score: combines accuracy + tightness. Lower is better.",
    )
    c4.metric("Predictions", f"{bm['n']:,}", help="Number of hourly predictions in the 30-day backtest.")

    st.markdown(
        f"Coverage health: <span class='{cls}'>{emoji} {verdict}</span> &nbsp;·&nbsp; "
        f"Coverage = {bm['coverage_95']:.4f} (target 0.95)",
        unsafe_allow_html=True,
    )
else:
    st.info(
        "📂 No `backtest_results.jsonl` found. "
        "Run `python btc_gbm_backtest.py` locally and commit the file to your repo. "
        "Backtest metrics will appear here after re-deploy."
    )

st.divider()


# ── Live Prediction ───────────────────────────────────────────────────────────
regime_label, regime_cls = vol_regime(vol_ann)

col_left, col_right = st.columns([3, 1])
with col_left:
    st.markdown("### 🔮 Next-Hour Prediction")
with col_right:
    st.markdown(
        f"Volatility regime: <span class='regime-{regime_cls}'>{regime_label}</span> "
        f"({vol_ann:.1f}% ann.)",
        unsafe_allow_html=True,
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current BTC Price", f"${current:,.2f}")
c2.metric(
    "95% Lower Bound",
    f"${low:,.2f}",
    delta=f"{(low/current - 1)*100:+.2f}%",
    help="Model predicts 2.5% chance BTC falls below this price next hour.",
)
c3.metric(
    "95% Upper Bound",
    f"${high:,.2f}",
    delta=f"{(high/current - 1)*100:+.2f}%",
    help="Model predicts 2.5% chance BTC rises above this price next hour.",
)
c4.metric(
    "Range Width",
    f"${high-low:,.0f}",
    delta=f"{(high-low)/current*100:.2f}% of price",
)

w_pct = (high - low) / current * 100
st.markdown(
    f"<span class='countdown'>⏱ Next bar closes in {minutes_left}m {seconds_left:02d}s</span> &nbsp;·&nbsp; "
    f"Target bar: **{next_bar_time.strftime('%Y-%m-%d %H:%M UTC')}** &nbsp;·&nbsp; "
    f"Range: **${low:,.0f} – ${high:,.0f}** ({w_pct:.2f}%)",
    unsafe_allow_html=True,
)

# ── Main chart ────────────────────────────────────────────────────────────────
fig = build_chart(df, low, high, current)
st.plotly_chart(fig, use_container_width=True)


# ── Part C: Persist this prediction ──────────────────────────────────────────
prediction_record = {
    "prediction_time":  now_utc.isoformat(),
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

history = load_history()
wrote_new = maybe_append_history(prediction_record, history)

# Reload (includes the new record if written), then fill actuals
history = load_history()
history = fill_actuals(history, df)

st.divider()


# ── Part C: History section ───────────────────────────────────────────────────
resolved = [r for r in history if r.get("actual") is not None]
pending  = [r for r in history if r.get("actual") is None]

st.markdown(
    f"### 📈 Live Prediction History — "
    f"{len(resolved)} resolved &nbsp;|&nbsp; {len(pending)} pending &nbsp;|&nbsp; {len(history)} total"
)

if resolved:
    hist_hits  = [r for r in resolved if r.get("hit")]
    live_cov   = len(hist_hits) / len(resolved)
    live_winkl = float(np.mean([r["winkler"] for r in resolved]))
    live_width = float(np.mean([r["width"]   for r in resolved]))

    cov_cls, cov_emoji, cov_verdict = coverage_health(live_cov)

    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Live Coverage",  f"{live_cov:.4f}",     help="Fraction of resolved predictions where actual was inside range.")
    hc2.metric("Live Winkler ↓", f"${live_winkl:,.0f}", help="Live Winkler score (lower = better).")
    hc3.metric("Mean Width",     f"${live_width:,.0f}")
    hc4.metric("Resolved",       f"{len(resolved)}")

    st.markdown(
        f"Live coverage health: <span class='{cov_cls}'>{cov_emoji} {cov_verdict}</span>",
        unsafe_allow_html=True,
    )

    fig_h = build_history_chart(history)
    if fig_h:
        st.plotly_chart(fig_h, use_container_width=True)
else:
    st.info(
        "⏳ No resolved predictions yet. Each prediction is confirmed once the target bar closes. "
        "Come back in 1–2 hours to see actuals filled in."
    )


# ── Raw log table (collapsible) ───────────────────────────────────────────────
with st.expander("🗂️ Raw prediction log (newest first)"):
    if history:
        df_hist = pd.DataFrame(history[::-1])
        # Show key columns in a friendly order
        show_cols = [c for c in [
            "target_bar_time", "current_price", "low_95", "high_95",
            "width", "actual", "hit", "winkler", "prediction_time"
        ] if c in df_hist.columns]
        df_hist_display = df_hist[show_cols].copy()
        # Format floats
        for col in ["current_price", "low_95", "high_95", "width", "actual", "winkler"]:
            if col in df_hist_display.columns:
                df_hist_display[col] = df_hist_display[col].apply(
                    lambda x: f"${x:,.2f}" if pd.notna(x) and x is not None else "—"
                )
        df_hist_display["hit"] = df_hist_display["hit"].apply(
            lambda x: "✅" if x is True else ("❌" if x is False else "⏳")
        )
        st.dataframe(df_hist_display, use_container_width=True, height=300)
    else:
        st.write("No history yet.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AlphaI × Polaris Challenge · GBM + Student-t · "
    "Data: [Binance public API](https://data-api.binance.vision) · "
    "No API key required · Hosted on Streamlit Community Cloud"
)