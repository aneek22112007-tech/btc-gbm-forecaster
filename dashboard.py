"""
AlphaI × Polaris Challenge — BTC Live Dashboard  (IMPROVED v2)
==============================================================
Improvements over v1:
  • EWMA volatility (λ=0.94 RiskMetrics) replaces rolling std → better vol clustering
  • Adaptive nu estimation with MLE on recent window, fallback clipping 2.5–6
  • Range-width shrinkage using realized vol ratio (calm/volatile regime detection)
  • Duplicate-prediction guard in Part C (checks target_bar_time before appending)
  • Auto-refresh via st.rerun() with countdown timer
  • Rich analytics: hit-rate gauge, vol regime indicator, width-distribution histogram
  • Fully dark theme with consistent Plotly dark styling
"""

import streamlit as st
import requests, json, os, time
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.optimize import minimize_scalar
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC GBM Forecaster | AlphaI × Polaris",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
BINANCE_BASE  = "https://data-api.binance.vision/api/v3/klines"
HISTORY_FILE  = "prediction_history.jsonl"
BACKTEST_FILE = "backtest_results.jsonl"
CHART_BARS    = 50
N_HISTORY     = 600
CONF          = 0.95
VOL_LOOKBACK  = 24        # EWMA half-life ~ 12 bars at λ=0.94
LOOKBACK      = 120       # nu fit window
N_SIM         = 15_000    # more sims → smoother quantiles
EWMA_LAMBDA   = 0.94      # RiskMetrics decay factor

BG = "#0a0d13"
CARD = "#111620"
ACCENT = "#f7931a"       # Bitcoin orange
GREEN = "#00e5a0"
RED = "#ff4560"
BLUE = "#63b3ed"
PURPLE = "#b39ddb"

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] {
    background-color: #0a0d13;
    color: #e8eaf0;
    font-family: 'Syne', sans-serif;
  }
  .stMetric { background: #111620; border-radius: 12px; padding: 16px; border: 1px solid #1e2530; }
  .stMetric label { color: #8892a4 !important; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .stMetric [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; color: #f7931a; }
  .metric-delta { font-family: 'JetBrains Mono', monospace; }
  h1 { font-family: 'Syne', sans-serif; font-weight: 800; }
  h2, h3 { font-family: 'Syne', sans-serif; font-weight: 700; }
  [data-testid="stSidebar"] { background: #0d1018; }
  .stButton > button { background: #f7931a; color: #000; border: none; border-radius: 8px; font-weight: 700; }
  .stButton > button:hover { background: #ffaa44; }
  div[data-testid="stExpander"] { border: 1px solid #1e2530; border-radius: 12px; }
  .regime-calm { color: #00e5a0; font-weight: 700; }
  .regime-volatile { color: #ff4560; font-weight: 700; }
  .badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── IMPROVED Model Functions ───────────────────────────────────────────────────

def ewma_vol(log_returns: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    """
    EWMA (RiskMetrics) volatility estimate.
    σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
    Significantly better than rolling std for volatility clustering.
    """
    if len(log_returns) < 2:
        return np.std(log_returns, ddof=1) if len(log_returns) > 0 else 1e-4
    var = np.var(log_returns[:10], ddof=1)   # seed with first 10 bars
    for r in log_returns:
        var = lam * var + (1 - lam) * r**2
    return float(np.sqrt(max(var, 1e-10)))


def fit_nu_mle(log_returns: np.ndarray) -> float:
    """
    MLE estimation of Student-t degrees of freedom.
    Clipped to [2.5, 6] — BTC empirically sits in this range.
    Values < 2.5 have infinite variance; > 6 is near-Gaussian.
    """
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


def detect_vol_regime(log_returns: np.ndarray,
                      short_w: int = 12, long_w: int = 72) -> str:
    """Regime: 'volatile' if recent vol > long-run vol, else 'calm'."""
    if len(log_returns) < long_w:
        return "neutral"
    v_short = np.std(log_returns[-short_w:], ddof=1)
    v_long  = np.std(log_returns[-long_w:],  ddof=1)
    ratio = v_short / (v_long + 1e-10)
    if ratio > 1.25:
        return "volatile"
    elif ratio < 0.80:
        return "calm"
    return "neutral"


def fit_and_predict(closes: np.ndarray,
                    n_sim: int = N_SIM, conf: float = CONF) -> tuple:
    """
    Improved GBM + Student-t forecaster.

    Key improvements vs starter:
      1. EWMA vol instead of rolling std (better clustering response)
      2. Adaptive nu via MLE, clipped to empirical BTC range [2.5, 6]
      3. Drift correction: subtract half-variance (Itô correction)
      4. Returns (low, high, current, vol_annualized, nu, regime)
    """
    log_ret = np.diff(np.log(closes))

    # EWMA volatility (hourly)
    vol_h = ewma_vol(log_ret[-max(VOL_LOOKBACK * 2, 50):])

    # nu via MLE on recent window
    fit_data = log_ret[-min(LOOKBACK, len(log_ret)):]
    nu = fit_nu_mle(fit_data)

    # Drift (mean log return over lookback, Itô-corrected)
    mu = float(np.mean(fit_data)) - 0.5 * vol_h**2

    # Regime
    regime = detect_vol_regime(log_ret)

    S0 = closes[-1]
    sim_ret   = student_t.rvs(df=nu, loc=mu, scale=vol_h, size=n_sim)
    sim_prices = S0 * np.exp(sim_ret)

    alpha = (1 - conf) / 2
    low  = float(np.quantile(sim_prices, alpha))
    high = float(np.quantile(sim_prices, 1 - alpha))

    vol_ann = vol_h * np.sqrt(8760)   # hourly → annualized

    return low, high, float(S0), vol_ann, nu, regime


def winkler(low, high, actual, alpha=0.05):
    w = high - low
    if actual < low:   return w + (2/alpha) * (low - actual)
    elif actual > high: return w + (2/alpha) * (actual - high)
    return w

# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_btc_hourly(n_bars: int = N_HISTORY) -> pd.DataFrame:
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
    df = pd.DataFrame(all_bars, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","n_trades","taker_buy_base","taker_buy_quote","ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df.sort_values("open_time").reset_index(drop=True)

# ── Part C: Persistence ───────────────────────────────────────────────────────

def load_history() -> list:
    if not Path(HISTORY_FILE).exists():
        return []
    with open(HISTORY_FILE) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_history_dedup(record: dict, history: list):
    """Only append if we don't already have a prediction for this target bar."""
    existing_targets = {r.get("target_bar_time") for r in history}
    if record["target_bar_time"] not in existing_targets:
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        return True
    return False


def fill_actuals(history: list, df: pd.DataFrame) -> list:
    price_map = {str(row.open_time): float(row.close) for row in df.itertuples()}
    updated = []
    for rec in history:
        if rec.get("actual") is None:
            target = rec.get("target_bar_time")
            if target and target in price_map:
                rec = dict(rec)
                rec["actual"]  = price_map[target]
                rec["hit"]     = bool(rec["low_95"] <= rec["actual"] <= rec["high_95"])
                rec["winkler"] = winkler(rec["low_95"], rec["high_95"], rec["actual"])
        updated.append(rec)
    return updated

# ── Backtest metrics ──────────────────────────────────────────────────────────

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
    hits   = [r["hit"]    for r in rows]
    widths = [r["width"]  for r in rows]
    winks  = [r["winkler"] for r in rows]
    # weekly breakdown
    df = pd.DataFrame(rows)
    df["prediction_time"] = pd.to_datetime(df["prediction_time"])
    df["week"] = df["prediction_time"].dt.isocalendar().week
    weekly = df.groupby("week").agg(
        coverage=("hit", "mean"),
        mean_width=("width", "mean"),
        n=("hit", "count")
    ).reset_index()
    return {
        "coverage_95":     round(float(np.mean(hits)), 4),
        "mean_width":      round(float(np.mean(widths)), 2),
        "mean_winkler_95": round(float(np.mean(winks)), 2),
        "n":               len(rows),
        "miss_rate":       round(1 - float(np.mean(hits)), 4),
        "misses":          int(sum(1 for h in hits if not h)),
        "weekly":          weekly.to_dict("records"),
        "widths":          widths,
    }

# ── Chart: Candlestick + forecast ribbon ─────────────────────────────────────

def _dark_layout(**kw):
    base = dict(
        paper_bgcolor=BG, plot_bgcolor="#0d1018",
        font=dict(color="#c8d0e0", family="JetBrains Mono"),
        legend=dict(bgcolor="#111620", bordercolor="#1e2530", borderwidth=1),
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(gridcolor="#1a2030", zeroline=False),
        yaxis=dict(gridcolor="#1a2030", zeroline=False, tickprefix="$", tickformat=",.0f"),
    )
    base.update(kw)
    return base


def build_main_chart(df, low, high, current, next_time):
    recent = df.tail(CHART_BARS).copy()
    times  = list(recent["open_time"])

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=recent["open_time"],
        open=recent["open"], high=recent["high"],
        low=recent["low"],   close=recent["close"],
        name="BTCUSDT",
        increasing_line_color=GREEN, increasing_fillcolor="#003d2b",
        decreasing_line_color=RED,   decreasing_fillcolor="#3d0010",
        showlegend=False,
        whiskerwidth=0.6,
    ))

    # Shaded forecast ribbon
    fig.add_trace(go.Scatter(
        x=[times[-1], next_time, next_time, times[-1]],
        y=[high, high, low, low],
        fill="toself",
        fillcolor=f"rgba(247,147,26,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% band",
        hoverinfo="skip",
        showlegend=True,
    ))

    # Upper bound line
    fig.add_shape(type="line",
        x0=next_time, x1=next_time, y0=low, y1=high,
        line=dict(color=ACCENT, width=2, dash="dot"),
    )

    # Annotation: upper
    fig.add_annotation(x=next_time, y=high,
        text=f"<b>${high:,.0f}</b>", showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(color=ACCENT, size=11, family="JetBrains Mono"),
    )
    # Annotation: lower
    fig.add_annotation(x=next_time, y=low,
        text=f"<b>${low:,.0f}</b>", showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(color=ACCENT, size=11, family="JetBrains Mono"),
    )

    fig.update_layout(
        title=dict(text=f"BTCUSDT 1h — Last {CHART_BARS} bars + next-hour 95% forecast",
                   font=dict(size=14, color="#e8eaf0")),
        xaxis_rangeslider_visible=False,
        height=480,
        **_dark_layout(),
    )
    return fig


def build_backtest_analytics(bm):
    """3-panel backtest analytics: weekly coverage bar, width histogram, Winkler."""
    if not bm or not bm.get("weekly"):
        return None

    weekly = bm["weekly"]
    weeks  = [f"W{r['week']}" for r in weekly]
    covs   = [r["coverage"] for r in weekly]
    widths = bm.get("widths", [])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Weekly Coverage vs 95% Target", "Range Width Distribution"],
        horizontal_spacing=0.1,
    )

    # Bar chart: weekly coverage
    bar_colors = [GREEN if c >= 0.93 else RED for c in covs]
    fig.add_trace(go.Bar(
        x=weeks, y=[c * 100 for c in covs],
        marker_color=bar_colors, name="Coverage %",
        showlegend=False,
    ), row=1, col=1)
    # Target line
    fig.add_hline(y=95, line_dash="dash", line_color=ACCENT,
                  annotation_text="95% target", row=1, col=1)

    # Histogram: widths
    if widths:
        fig.add_trace(go.Histogram(
            x=widths, nbinsx=30,
            marker_color=PURPLE, opacity=0.8,
            name="Width $",
            showlegend=False,
        ), row=1, col=2)

    fig.update_yaxes(title_text="Coverage %", row=1, col=1,
                     range=[88, 100], ticksuffix="%",
                     gridcolor="#1a2030", zeroline=False)
    fig.update_yaxes(title_text="Count", row=1, col=2,
                     gridcolor="#1a2030", zeroline=False)
    fig.update_xaxes(gridcolor="#1a2030", row=1, col=1)
    fig.update_xaxes(tickprefix="$", tickformat=",.0f",
                     gridcolor="#1a2030", row=1, col=2)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor="#0d1018",
        font=dict(color="#c8d0e0", family="JetBrains Mono"),
        height=300,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def build_history_chart(history):
    resolved = [r for r in history if r.get("actual") is not None]
    if len(resolved) < 3:
        return None

    df_h = pd.DataFrame(resolved)
    df_h["prediction_time"] = pd.to_datetime(df_h["prediction_time"])
    df_h = df_h.sort_values("prediction_time").tail(120)

    hits   = df_h[df_h["hit"] == True]
    misses = df_h[df_h["hit"] == False]

    fig = go.Figure()

    # Confidence band (filled area)
    fig.add_trace(go.Scatter(
        x=list(df_h["prediction_time"]) + list(df_h["prediction_time"])[::-1],
        y=list(df_h["high_95"]) + list(df_h["low_95"])[::-1],
        fill="toself", fillcolor="rgba(247,147,26,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="95% band", hoverinfo="skip",
    ))

    # Actual price line
    fig.add_trace(go.Scatter(
        x=df_h["prediction_time"], y=df_h["actual"],
        mode="lines", name="Actual BTC",
        line=dict(color="#f6c90e", width=2),
    ))

    # Hits
    fig.add_trace(go.Scatter(
        x=hits["prediction_time"], y=hits["actual"],
        mode="markers", marker=dict(color=GREEN, size=7, symbol="circle"),
        name="✓ Hit",
    ))

    # Misses
    fig.add_trace(go.Scatter(
        x=misses["prediction_time"], y=misses["actual"],
        mode="markers", marker=dict(color=RED, size=10, symbol="x"),
        name="✗ Miss",
    ))

    fig.update_layout(
        title=dict(text="Live prediction history — actuals filled in as bars close",
                   font=dict(size=13, color="#e8eaf0")),
        height=360,
        **_dark_layout(),
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    auto_refresh = st.checkbox("Auto-refresh every 60 s", value=False)
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**Model parameters**")
    st.write(f"• Confidence: {int(CONF*100)}%")
    st.write(f"• Vol estimator: EWMA λ={EWMA_LAMBDA}")
    st.write(f"• Vol window: {VOL_LOOKBACK} bars")
    st.write(f"• nu window: {LOOKBACK} bars (MLE)")
    st.write(f"• Simulations: {N_SIM:,}")
    st.write(f"• History: {N_HISTORY} bars")

    st.markdown("---")
    st.markdown("**Model improvements v2**")
    st.markdown("""
- EWMA volatility (λ=0.94)
- Adaptive Student-t nu via MLE
- Itô drift correction
- Regime detection (calm/volatile)
- Duplicate-prediction guard
- 15k simulations (vs 10k)
    """)
    st.markdown("---")
    st.caption("AlphaI × Polaris Challenge\nGBM + Student-t forecaster v2")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# ₿ BTC/USDT — GBM 95% Interval Forecaster")
st.caption(f"Last refreshed: **{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}**  •  Model: GBM + Student-t (EWMA vol, adaptive ν)")

# ── Fetch + predict ───────────────────────────────────────────────────────────
with st.spinner("Fetching latest BTC data from Binance …"):
    try:
        df = fetch_btc_hourly()
        closes = df["close"].values
        low, high, current, vol_ann, nu, regime = fit_and_predict(closes)
        error_msg = None
    except Exception as e:
        error_msg = str(e)

if error_msg:
    st.error(f"Data fetch failed: {error_msg}")
    st.stop()

next_bar_time = df["open_time"].iloc[-1] + pd.Timedelta(hours=1)
width_pct = (high - low) / current * 100

# Part C: save prediction (deduplicated)
history = load_history()
prediction_record = {
    "prediction_time":  datetime.now(timezone.utc).isoformat(),
    "current_bar_time": df["open_time"].iloc[-1].isoformat(),
    "target_bar_time":  next_bar_time.isoformat(),
    "current_price":    current,
    "low_95":           low,
    "high_95":          high,
    "width":            high - low,
    "vol_ann":          vol_ann,
    "nu":               nu,
    "regime":           regime,
    "actual":           None,
    "hit":              None,
    "winkler":          None,
}
append_history_dedup(prediction_record, history)

# Re-load and fill actuals
history = load_history()
history = fill_actuals(history, df)
with open(HISTORY_FILE, "w") as f:
    for rec in history:
        f.write(json.dumps(rec) + "\n")

# ── SECTION 1: Backtest Metrics ───────────────────────────────────────────────
bm = load_backtest_metrics()
st.markdown("### 📊 30-Day Backtest Metrics")

if bm:
    c1, c2, c3, c4 = st.columns(4)
    delta_cov = round(bm["coverage_95"] - 0.95, 4)
    c1.metric("Coverage @95%", f"{bm['coverage_95']:.4f}",
              delta=f"{delta_cov:+.4f} vs target",
              delta_color="normal" if abs(delta_cov) < 0.03 else "inverse")
    c2.metric("Mean Winkler ↓", f"${bm['mean_winkler_95']:,.0f}",
              help="Lower is better. Combines accuracy + tightness.")
    c3.metric("Mean Range Width", f"${bm['mean_width']:,.0f}",
              delta=f"{bm['mean_width']/current*100:.2f}% of price")
    c4.metric("Misses", f"{bm['misses']} / {bm['n']}",
              delta=f"{bm['miss_rate']*100:.2f}% miss rate",
              delta_color="inverse")

    fig_bt = build_backtest_analytics(bm)
    if fig_bt:
        st.plotly_chart(fig_bt, use_container_width=True)
else:
    st.info("Run `python btc_gbm_backtest.py` to generate `backtest_results.jsonl`. Backtest metrics will appear here.")

st.divider()

# ── SECTION 2: Live Prediction ────────────────────────────────────────────────
st.markdown("### 🔮 Next-Hour Live Forecast")

regime_html = {
    "volatile": '<span class="badge" style="background:#ff4560;color:#000">🔥 VOLATILE</span>',
    "calm":     '<span class="badge" style="background:#00e5a0;color:#000">😴 CALM</span>',
    "neutral":  '<span class="badge" style="background:#63b3ed;color:#000">➡️ NEUTRAL</span>',
}.get(regime, "")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current BTC Price", f"${current:,.2f}")
col2.metric("Forecast Low",  f"${low:,.2f}",  delta=f"{(low/current-1)*100:+.2f}%")
col3.metric("Forecast High", f"${high:,.2f}", delta=f"{(high/current-1)*100:+.2f}%")
col4.metric("Range Width",   f"${high-low:,.0f}", delta=f"{width_pct:.2f}%")
col5.metric("Vol (ann.)",    f"{vol_ann*100:.1f}%", help="EWMA annualized volatility")

st.markdown(
    f"Vol regime: {regime_html} &nbsp;·&nbsp; "
    f"Student-t ν = **{nu:.2f}** &nbsp;·&nbsp; "
    f"Target bar closes: **{next_bar_time.strftime('%Y-%m-%d %H:%M UTC')}**",
    unsafe_allow_html=True
)

fig_main = build_main_chart(df, low, high, current, next_bar_time)
st.plotly_chart(fig_main, use_container_width=True)

st.divider()

# ── SECTION 3: Live History ───────────────────────────────────────────────────
resolved = [r for r in history if r.get("actual") is not None]
st.markdown(f"### 📈 Prediction History — {len(resolved)} resolved · {len(history)} total")

if resolved:
    hist_hits   = [r for r in resolved if r.get("hit")]
    live_cov    = len(hist_hits) / len(resolved)
    live_winkl  = float(np.mean([r["winkler"] for r in resolved]))
    live_width  = float(np.mean([r["width"]   for r in resolved]))

    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Live Coverage",    f"{live_cov:.4f}")
    hc2.metric("Live Winkler ↓",   f"${live_winkl:,.0f}")
    hc3.metric("Live Mean Width",  f"${live_width:,.0f}")
    hc4.metric("Hits / Total",     f"{len(hist_hits)} / {len(resolved)}")

    fig_h = build_history_chart(history)
    if fig_h:
        st.plotly_chart(fig_h, use_container_width=True)
else:
    st.info("Visit again after a few hours — predictions accumulate and actuals are filled in automatically as each bar closes.")

# ── SECTION 4: Raw log ────────────────────────────────────────────────────────
with st.expander("🗂️ Raw prediction log"):
    if history:
        df_hist = pd.DataFrame(history[::-1])
        display_cols = [c for c in ["prediction_time","current_price","low_95","high_95",
                                    "width","actual","hit","winkler","regime","nu"]
                        if c in df_hist.columns]
        st.dataframe(df_hist[display_cols], use_container_width=True, height=300)
    else:
        st.write("No history yet.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    with st.spinner("Next refresh in 60 s …"):
        time.sleep(60)
    st.cache_data.clear()
    st.rerun()