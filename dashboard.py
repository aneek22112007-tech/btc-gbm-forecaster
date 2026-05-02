"""
AlphaI × Polaris — BTC GBM Forecaster  |  Premium Dashboard v2
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

st.set_page_config(
    page_title="BTC Forecaster · AlphaI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #080a0f; color: #e2e8f0; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="metric-container"] {
    background: #0d1117; border: 1px solid #1e2a3a;
    border-radius: 12px; padding: 1.2rem 1.4rem !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #f59e0b; }
[data-testid="metric-container"] label {
    font-family: 'Space Mono', monospace !important; font-size: 10px !important;
    letter-spacing: 0.12em !important; color: #4a6080 !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important; font-size: 1.5rem !important;
    color: #f8fafc !important; font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'Space Mono', monospace !important; font-size: 11px !important;
}
[data-testid="stExpander"] {
    background: #0d1117 !important; border: 1px solid #1e2a3a !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important; border-right: 1px solid #1e2a3a !important;
}
.stTabs [data-baseweb="tab-list"] { background: #0d1117; border-radius: 8px; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace; font-size: 11px; color: #4a6080;
    background: transparent; border-radius: 6px;
}
.stTabs [aria-selected="true"] { color: #f59e0b !important; background: #1e2a3a !important; }
hr { border-color: #1e2a3a !important; margin: 1.5rem 0 !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080a0f; }
::-webkit-scrollbar-thumb { background: #1e2a3a; border-radius: 2px; }

.price-hero {
    font-family: 'Space Mono', monospace; font-size: 3rem; font-weight: 700;
    color: #f8fafc; letter-spacing: -0.02em; line-height: 1; margin: 0.4rem 0;
}
.range-card {
    background: linear-gradient(135deg, #0d1117 0%, #0a1628 100%);
    border: 1px solid #1e3a5f; border-radius: 14px; padding: 1.4rem 1.6rem;
}
.range-label {
    font-family: 'Space Mono', monospace; font-size: 9px;
    letter-spacing: 0.15em; color: #4a6080; text-transform: uppercase; margin-bottom: 6px;
}
.range-value {
    font-family: 'Space Mono', monospace; font-size: 1.5rem;
    font-weight: 700; color: #60a5fa;
}
.section-tag {
    font-family: 'Space Mono', monospace; font-size: 9px;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: #f59e0b; margin-bottom: 0.8rem; display: block;
}
.width-bar-bg {
    background: #1e2a3a; border-radius: 4px; height: 5px;
    margin: 10px 0 4px; overflow: hidden;
}
.width-bar-fill {
    background: linear-gradient(90deg, #f59e0b, #f97316);
    height: 100%; border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BINANCE_BASE  = "https://data-api.binance.vision/api/v3/klines"
HISTORY_FILE  = "prediction_history.jsonl"
BACKTEST_FILE = "backtest_results.jsonl"
CHART_BARS    = 60
N_HISTORY     = 500
CONF          = 0.95
VOL_LOOKBACK  = 20
LOOKBACK      = 100
N_SIM         = 10_000
PLOT_BG       = "#080a0f"
GRID_COLOR    = "#1e2a3a"
TEXT_COLOR    = "#94a3b8"
FONT_MONO     = "Space Mono"


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=55)
def fetch_btc_hourly(n_bars=N_HISTORY):
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


def fit_and_predict(closes):
    log_ret    = np.diff(np.log(closes))
    recent_vol = np.std(log_ret[-VOL_LOOKBACK:], ddof=1)
    fit_data   = log_ret[-min(LOOKBACK, len(log_ret)):]
    nu, mu_t, _ = student_t.fit(fit_data, floc=0)
    S0         = closes[-1]
    sim_ret    = student_t.rvs(df=nu, loc=mu_t, scale=recent_vol, size=N_SIM)
    sim_prices = S0 * np.exp(sim_ret)
    a          = (1 - CONF) / 2
    return float(np.quantile(sim_prices, a)), float(np.quantile(sim_prices, 1-a)), float(S0)


def winkler_score(low, high, actual, alpha=0.05):
    w = high - low
    if actual < low:    return w + (2/alpha)*(low - actual)
    elif actual > high: return w + (2/alpha)*(actual - high)
    return w


def load_history():
    if not Path(HISTORY_FILE).exists(): return []
    with open(HISTORY_FILE) as f:
        return [json.loads(l) for l in f if l.strip()]

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        for rec in history:
            f.write(json.dumps(rec) + "\n")

def fill_actuals(history, df):
    price_map = {str(row.open_time): row.close for row in df.itertuples()}
    updated = []
    for rec in history:
        if rec.get("actual") is None:
            target = rec.get("target_bar_time")
            if target and target in price_map:
                rec = dict(rec)
                rec["actual"]  = price_map[target]
                rec["hit"]     = rec["low_95"] <= rec["actual"] <= rec["high_95"]
                rec["winkler"] = winkler_score(rec["low_95"], rec["high_95"], rec["actual"])
        updated.append(rec)
    return updated


@st.cache_data
def load_backtest_metrics():
    if not Path(BACKTEST_FILE).exists(): return None
    rows = []
    with open(BACKTEST_FILE) as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    if not rows: return None
    return {
        "coverage":     round(float(np.mean([r["hit"] for r in rows])), 4),
        "mean_width":   round(float(np.mean([r["width"] for r in rows])), 0),
        "mean_winkler": round(float(np.mean([r["winkler"] for r in rows])), 0),
        "n":            len(rows),
        "misses":       len([r for r in rows if not r["hit"]]),
        "rows":         rows,
    }


# ── Chart builders ─────────────────────────────────────────────────────────────
def build_main_chart(df, low, high, current):
    recent    = df.tail(CHART_BARS).copy()
    next_time = recent["open_time"].iloc[-1] + pd.Timedelta(hours=1)
    last_time = recent["open_time"].iloc[-1]

    fig = go.Figure()

    # Forecast zone
    fig.add_trace(go.Scatter(
        x=[last_time, next_time, next_time, last_time],
        y=[high, high, low, low],
        fill="toself", fillcolor="rgba(245,158,11,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=recent["open_time"],
        open=recent["open"], high=recent["high"],
        low=recent["low"],   close=recent["close"],
        increasing=dict(line=dict(color="#10b981", width=1.2), fillcolor="#10b981"),
        decreasing=dict(line=dict(color="#ef4444", width=1.2), fillcolor="#ef4444"),
        showlegend=False,
    ))

    # Dotted lines from last close to bounds
    for bound in [high, low]:
        fig.add_trace(go.Scatter(
            x=[last_time, next_time], y=[current, bound],
            mode="lines", line=dict(color="#f59e0b", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    # Bound markers
    for val, label in [(high, f"▲ ${high:,.0f}"), (low, f"▼ ${low:,.0f}")]:
        fig.add_trace(go.Scatter(
            x=[next_time], y=[val],
            mode="markers+text",
            marker=dict(color="#f59e0b", size=9, symbol="diamond"),
            text=[f"  {label}"],
            textposition="middle right",
            textfont=dict(family=FONT_MONO, size=11, color="#f59e0b"),
            showlegend=False,
        ))

    # Annotations
    fig.add_vline(x=last_time.timestamp()*1000,
                  line=dict(color="#1e2a3a", width=1, dash="dot"))
    fig.add_annotation(x=last_time, y=recent["high"].max()*1.001,
        text="NOW", showarrow=False,
        font=dict(family=FONT_MONO, size=9, color="#4a6080"), xanchor="center")
    fig.add_annotation(x=next_time, y=recent["high"].max()*1.001,
        text="+1H", showarrow=False,
        font=dict(family=FONT_MONO, size=9, color="#f59e0b"), xanchor="center")

    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans", color=TEXT_COLOR),
        height=430, margin=dict(l=0, r=90, t=20, b=0),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   rangeslider=dict(visible=False),
                   tickfont=dict(family=FONT_MONO, size=10, color="#4a6080"),
                   showline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   tickprefix="$", tickformat=",.0f", side="right",
                   tickfont=dict(family=FONT_MONO, size=10, color="#4a6080"),
                   showline=False),
        hoverlabel=dict(bgcolor="#0d1117", bordercolor="#1e2a3a",
                        font=dict(family=FONT_MONO, size=11, color="#e2e8f0")),
    )
    return fig


def build_backtest_chart(rows):
    df_b = pd.DataFrame(rows)
    df_b["prediction_time"] = pd.to_datetime(df_b["prediction_time"])
    df_b = df_b.sort_values("prediction_time")
    hits   = df_b[df_b["hit"] == True]
    misses = df_b[df_b["hit"] == False]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(df_b["prediction_time"]) + list(df_b["prediction_time"])[::-1],
        y=list(df_b["high"]) + list(df_b["low"])[::-1],
        fill="toself", fillcolor="rgba(245,158,11,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="95% band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df_b["prediction_time"], y=df_b["actual"],
        mode="lines", line=dict(color="#60a5fa", width=1.5), name="Actual BTC",
    ))
    fig.add_trace(go.Scatter(
        x=hits["prediction_time"][::4], y=hits["actual"][::4],
        mode="markers", marker=dict(color="#10b981", size=4, opacity=0.5), name="Hit",
    ))
    fig.add_trace(go.Scatter(
        x=misses["prediction_time"], y=misses["actual"],
        mode="markers",
        marker=dict(color="#ef4444", size=10, symbol="x", line=dict(width=2, color="#ef4444")),
        name="Miss",
        customdata=np.column_stack([misses["low"].round(0), misses["high"].round(0), misses["winkler"].round(0)]),
        hovertemplate="<b>MISS</b><br>Actual: $%{y:,.0f}<br>Range: $%{customdata[0]:,.0f}–$%{customdata[1]:,.0f}<br>Winkler: $%{customdata[2]:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans", color=TEXT_COLOR),
        height=300, margin=dict(l=0, r=60, t=10, b=0),
        legend=dict(bgcolor="#0d1117", bordercolor="#1e2a3a", borderwidth=1,
                    font=dict(family=FONT_MONO, size=10), orientation="h", y=1.08, x=0),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   tickfont=dict(family=FONT_MONO, size=10, color="#4a6080"), showline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   tickprefix="$", tickformat=",.0f", side="right",
                   tickfont=dict(family=FONT_MONO, size=10, color="#4a6080"), showline=False),
        hoverlabel=dict(bgcolor="#0d1117", bordercolor="#1e2a3a",
                        font=dict(family=FONT_MONO, size=11)),
    )
    return fig


def build_width_chart(rows):
    df_b = pd.DataFrame(rows)
    df_b["prediction_time"] = pd.to_datetime(df_b["prediction_time"])
    df_b = df_b.sort_values("prediction_time")
    misses = df_b[df_b["hit"] == False]
    mean_w = df_b["width"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_b["prediction_time"], y=df_b["width"],
        mode="lines", fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
        line=dict(color="#f59e0b", width=1.2), name="Range width",
    ))
    fig.add_trace(go.Scatter(
        x=misses["prediction_time"], y=misses["width"],
        mode="markers", marker=dict(color="#ef4444", size=8), name="Miss",
    ))
    fig.add_hline(y=mean_w, line=dict(color="#4a6080", width=1, dash="dot"),
                  annotation_text=f"  avg ${mean_w:,.0f}",
                  annotation_font=dict(family=FONT_MONO, size=9, color="#4a6080"))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans", color=TEXT_COLOR),
        height=180, margin=dict(l=0, r=60, t=10, b=0), showlegend=False,
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=False, zeroline=False,
                   tickfont=dict(family=FONT_MONO, size=10, color="#4a6080"), showline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   tickprefix="$", tickformat=",.0f", side="right",
                   tickfont=dict(family=FONT_MONO, size=10, color="#4a6080"), showline=False),
        hoverlabel=dict(bgcolor="#0d1117", bordercolor="#1e2a3a",
                        font=dict(family=FONT_MONO, size=11)),
    )
    return fig


def build_history_chart(history):
    resolved = [r for r in history if r.get("actual") is not None]
    if len(resolved) < 3: return None
    df_h = pd.DataFrame(resolved)
    df_h["prediction_time"] = pd.to_datetime(df_h["prediction_time"])
    df_h = df_h.sort_values("prediction_time").tail(120)
    hits   = df_h[df_h["hit"] == True]
    misses = df_h[df_h["hit"] == False]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(df_h["prediction_time"]) + list(df_h["prediction_time"])[::-1],
        y=list(df_h["high_95"]) + list(df_h["low_95"])[::-1],
        fill="toself", fillcolor="rgba(96,165,250,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="95% band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df_h["prediction_time"], y=df_h["actual"],
        mode="lines+markers", line=dict(color="#f8fafc", width=1.5),
        marker=dict(size=4, color="#f8fafc"), name="Actual BTC",
    ))
    fig.add_trace(go.Scatter(
        x=hits["prediction_time"], y=hits["actual"],
        mode="markers", marker=dict(color="#10b981", size=7), name="Hit",
    ))
    if len(misses):
        fig.add_trace(go.Scatter(
            x=misses["prediction_time"], y=misses["actual"],
            mode="markers",
            marker=dict(color="#ef4444", size=10, symbol="x", line=dict(width=2)),
            name="Miss",
        ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans", color=TEXT_COLOR),
        height=260, margin=dict(l=0, r=60, t=10, b=0),
        legend=dict(bgcolor="#0d1117", bordercolor="#1e2a3a", borderwidth=1,
                    font=dict(family=FONT_MONO, size=10), orientation="h", y=1.1, x=0),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   tickfont=dict(family=FONT_MONO, size=10, color="#4a6080"), showline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   tickprefix="$", tickformat=",.0f", side="right",
                   tickfont=dict(family=FONT_MONO, size=10, color="#4a6080"), showline=False),
        hoverlabel=dict(bgcolor="#0d1117", bordercolor="#1e2a3a",
                        font=dict(family=FONT_MONO, size=11)),
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Config")
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    if st.button("↺  Refresh now"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("""<div style='font-family:Space Mono,monospace;font-size:11px;
    color:#4a6080;line-height:2.2'>MODEL<br><span style='color:#94a3b8'>GBM + Student-t</span><br>
    CONFIDENCE<br><span style='color:#94a3b8'>95%</span><br>
    VOL WINDOW<br><span style='color:#94a3b8'>20 bars</span><br>
    SIMULATIONS<br><span style='color:#94a3b8'>10,000</span><br>
    HISTORY FED<br><span style='color:#94a3b8'>500 bars</span></div>""",
    unsafe_allow_html=True)


# ── Fetch data ─────────────────────────────────────────────────────────────────
with st.spinner(""):
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
prev_close    = df["close"].iloc[-2]
pct_change    = (current - prev_close) / prev_close * 100
width_pct     = (high - low) / current * 100

# Part C — save prediction
rec = {
    "prediction_time":  datetime.now(timezone.utc).isoformat(),
    "current_bar_time": df["open_time"].iloc[-1].isoformat(),
    "target_bar_time":  next_bar_time.isoformat(),
    "current_price": current,
    "low_95": low, "high_95": high, "width": high - low,
    "actual": None, "hit": None, "winkler": None,
}
history = load_history()
history.append(rec)
history = fill_actuals(history, df)
save_history(history)


# ══════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;
     border-bottom:1px solid #1e2a3a;padding-bottom:1.2rem;margin-bottom:1.8rem'>
  <div>
    <div style='font-family:Space Mono,monospace;font-size:9px;letter-spacing:0.2em;
         color:#4a6080;text-transform:uppercase;margin-bottom:5px'>
         AlphaI × Polaris Challenge</div>
    <div style='font-family:Space Mono,monospace;font-size:1.5rem;font-weight:700;
         color:#f59e0b;letter-spacing:-0.01em'>₿ BTC/USDT — GBM Interval Forecaster</div>
  </div>
  <div style='text-align:right'>
    <div style='font-family:Space Mono,monospace;font-size:9px;color:#4a6080;
         letter-spacing:0.1em'>LAST UPDATED</div>
    <div style='font-family:Space Mono,monospace;font-size:11px;color:#94a3b8;margin-top:3px'>
         {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</div>
    <div style='display:inline-flex;align-items:center;gap:5px;margin-top:6px;
         background:#0d1117;border:1px solid #1e2a3a;border-radius:20px;
         padding:3px 10px;font-family:Space Mono,monospace;font-size:9px;color:#4a6080'>
      <span style='width:5px;height:5px;border-radius:50%;background:#10b981;
            box-shadow:0 0 5px #10b981;display:inline-block'></span>LIVE
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Live prediction row ─────────────────────────────────
st.markdown('<span class="section-tag">◈ Live prediction — next 1 hour</span>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1.5])

chg_color = "#10b981" if pct_change >= 0 else "#ef4444"
chg_arrow = "▲" if pct_change >= 0 else "▼"

with c1:
    st.markdown(f"""
    <div style='background:#0d1117;border:1px solid #1e2a3a;border-radius:14px;
         padding:1.4rem 1.6rem;height:100%'>
      <div class='range-label'>Current BTC price</div>
      <div class='price-hero'>${current:,.2f}</div>
      <div style='font-family:Space Mono,monospace;font-size:13px;
           color:{chg_color};margin-top:6px'>{chg_arrow} {abs(pct_change):.2f}% vs prev close</div>
      <div style='font-family:Space Mono,monospace;font-size:9px;
           color:#2a3a4a;margin-top:5px'>BTCUSDT · 1H · BINANCE</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    lp = (low/current - 1)*100
    st.markdown(f"""
    <div class='range-card' style='height:100%'>
      <div class='range-label'>95% lower bound</div>
      <div class='range-value'>${low:,.2f}</div>
      <div style='font-family:Space Mono,monospace;font-size:12px;
           color:#ef4444;margin-top:6px'>{lp:.2f}%</div>
    </div>""", unsafe_allow_html=True)

with c3:
    hp = (high/current - 1)*100
    st.markdown(f"""
    <div class='range-card' style='height:100%'>
      <div class='range-label'>95% upper bound</div>
      <div class='range-value'>${high:,.2f}</div>
      <div style='font-family:Space Mono,monospace;font-size:12px;
           color:#10b981;margin-top:6px'>+{hp:.2f}%</div>
    </div>""", unsafe_allow_html=True)

with c4:
    bar_w = min(width_pct / 3 * 100, 100)
    st.markdown(f"""
    <div style='background:#0d1117;border:1px solid #1e2a3a;border-radius:14px;
         padding:1.4rem 1.6rem;height:100%'>
      <div class='range-label'>Range width</div>
      <div style='font-family:Space Mono,monospace;font-size:1.4rem;
           font-weight:700;color:#f59e0b'>${high-low:,.0f}</div>
      <div class='width-bar-bg'>
        <div class='width-bar-fill' style='width:{bar_w:.0f}%'></div>
      </div>
      <div style='font-family:Space Mono,monospace;font-size:11px;
           color:#4a6080'>{width_pct:.2f}% of price</div>
      <div style='font-family:Space Mono,monospace;font-size:9px;
           color:#2a3a4a;margin-top:5px'>TARGET → {next_bar_time.strftime("%H:%M UTC")}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.plotly_chart(build_main_chart(df, low, high, current),
                use_container_width=True, config={"displayModeBar": False})

st.markdown("<hr>", unsafe_allow_html=True)

# ── Backtest metrics ────────────────────────────────────
st.markdown('<span class="section-tag">◈ Backtest metrics — 30-day walk-forward (679 bars)</span>',
            unsafe_allow_html=True)

bm = load_backtest_metrics()
if bm:
    delta_cov = bm["coverage"] - 0.95
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Coverage @95%",  f"{bm['coverage']:.4f}",
              delta=f"{delta_cov:+.4f} vs target", delta_color="normal")
    m2.metric("Mean Winkler ↓", f"${bm['mean_winkler']:,.0f}",
              help="Lower = better. Penalises both width and misses.")
    m3.metric("Mean Width",     f"${bm['mean_width']:,.0f}")
    m4.metric("Predictions",    f"{bm['n']:,}")
    m5.metric("Misses",         f"{bm['misses']}",
              delta=f"{bm['misses']/bm['n']*100:.1f}% miss rate",
              delta_color="inverse")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["  Price + 95% band  ", "  Range width over time  "])
    with tab1:
        st.plotly_chart(build_backtest_chart(bm["rows"]),
                        use_container_width=True, config={"displayModeBar": False})
    with tab2:
        st.markdown("""<div style='font-family:Space Mono,monospace;font-size:10px;
        color:#4a6080;margin-bottom:0.5rem'>
        Volatility clustering — model widens range during violent periods automatically
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(build_width_chart(bm["rows"]),
                        use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Commit `backtest_results.jsonl` to your repo to see metrics here.")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Live history (Part C) ────────────────────────────────
resolved = [r for r in history if r.get("actual") is not None]
st.markdown(
    f'<span class="section-tag">◈ Live prediction history — {len(resolved)} resolved · {len(history)} total</span>',
    unsafe_allow_html=True)

if resolved:
    live_cov   = np.mean([r["hit"] for r in resolved])
    live_winkl = np.mean([r["winkler"] for r in resolved])
    lh1, lh2, lh3 = st.columns(3)
    lh1.metric("Live Coverage",  f"{live_cov:.4f}")
    lh2.metric("Live Winkler ↓", f"${live_winkl:,.0f}")
    lh3.metric("Resolved",       len(resolved))
    fig_h = build_history_chart(history)
    if fig_h:
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

    with st.expander("  Prediction log (last 30)  "):
        recent_recs = sorted(history, key=lambda r: r["prediction_time"], reverse=True)[:30]
        rows_d = []
        for r in recent_recs:
            rows_d.append({
                "Time (UTC)": r["prediction_time"][:16].replace("T"," "),
                "Price":  f"${r['current_price']:,.0f}",
                "Lower":  f"${r['low_95']:,.0f}",
                "Upper":  f"${r['high_95']:,.0f}",
                "Width":  f"${r['width']:,.0f}",
                "Actual": f"${r['actual']:,.0f}" if r.get("actual") else "pending",
                "Result": "✓ HIT" if r.get("hit") is True else ("✗ MISS" if r.get("hit") is False else "—"),
            })
        st.dataframe(pd.DataFrame(rows_d), use_container_width=True,
                     hide_index=True, height=300)
else:
    st.markdown("""
    <div style='background:#0d1117;border:1px dashed #1e2a3a;border-radius:12px;
         padding:2rem;text-align:center;color:#4a6080;
         font-family:Space Mono,monospace;font-size:12px'>
    Visit again after an hour — predictions accumulate automatically as bars close.
    </div>""", unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────
st.markdown(f"""
<div style='border-top:1px solid #1e2a3a;margin-top:3rem;padding-top:1.2rem;
     font-family:Space Mono,monospace;font-size:9px;color:#2a3a4a;
     display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px'>
  <span>AlphaI × Polaris Challenge · GBM Forecaster v2</span>
  <span>Data: Binance BTCUSDT 1H · Model: GBM + Student-t · Confidence: 95%</span>
</div>
""", unsafe_allow_html=True)

if auto_refresh:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()