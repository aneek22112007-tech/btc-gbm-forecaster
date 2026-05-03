"""
BTC/USDT Next-Hour 95% Interval Forecaster
AlphaI x Polaris Challenge — Aneek Das

Parts B + C: live dashboard with prediction history

Run locally:  streamlit run dashboard.py
Deploy:       push to GitHub, connect on share.streamlit.io
"""

import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# non-blocking auto-refresh (optional dep)
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ---------- config ----------
st.set_page_config(
    page_title="BTC Forecaster",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- constants ----------
API_URL       = "https://data-api.binance.vision/api/v3/klines"  # no geo-block in India
HIST_FILE     = "prediction_history.jsonl"
BACKTEST_FILE = "backtest_results.jsonl"
N_BARS        = 500    # bars fed into model
CHART_BARS    = 50     # bars shown on chart
VOL_WINDOW    = 20     # short window for vol clustering
FIT_WINDOW    = 100    # longer window for Student-t fit
N_SIM         = 10000  # MC simulations
ALPHA         = 0.05   # 95% CI → alpha = 0.05


# ---------- data fetch ----------
@st.cache_data(ttl=60)
def get_btc_data(n=N_BARS):
    bars, end = [], None
    while len(bars) < n:
        params = {"symbol": "BTCUSDT", "interval": "1h", "limit": min(1000, n - len(bars))}
        if end:
            params["endTime"] = end
        r = requests.get(API_URL, params=params, timeout=15)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        bars = chunk + bars
        end = chunk[0][0] - 1
        if len(chunk) < 1000:
            break

    df = pd.DataFrame(bars, columns=[
        "ts", "open", "high", "low", "close", "vol",
        "close_ts", "qvol", "ntrades", "tbbase", "tbquote", "ignore"
    ])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)

    if len(df) < n * 0.85:
        st.warning(f"Only got {len(df)} bars (wanted {n}). Binance may be throttling.")
    return df


# ---------- model ----------
def predict(closes):
    """
    GBM + Student-t one-step-ahead 95% interval.
    Key ideas:
      - vol from last VOL_WINDOW bars only (captures clustering)
      - Student-t fit for fat tails (BTC blows up more than Gaussian expects)
      - simulate N_SIM paths, read off 2.5 / 97.5 percentiles
    """
    log_ret = np.diff(np.log(closes))

    # recent vol — short window so we react fast to regime changes
    recent_vol = np.std(log_ret[-VOL_WINDOW:], ddof=1)

    # fit Student-t on longer history to get degrees-of-freedom (nu)
    fit_data = log_ret[-min(FIT_WINDOW, len(log_ret)):]
    nu, mu, _ = student_t.fit(fit_data, floc=0)

    S0 = closes[-1]
    sim_ret    = student_t.rvs(df=nu, loc=mu, scale=recent_vol, size=N_SIM)
    sim_prices = S0 * np.exp(sim_ret)

    low  = float(np.percentile(sim_prices, 2.5))
    high = float(np.percentile(sim_prices, 97.5))

    # annualised vol just for display
    vol_ann = recent_vol * np.sqrt(8760) * 100
    return low, high, float(S0), vol_ann


# ---------- winkler score ----------
def winkler(low, high, actual):
    w = high - low
    if actual < low:
        return w + (2 / ALPHA) * (low - actual)
    elif actual > high:
        return w + (2 / ALPHA) * (actual - high)
    return w


# ---------- vol regime ----------
def regime(vol_ann):
    if vol_ann < 40:
        return "🟢 Calm", "green"
    elif vol_ann < 80:
        return "🟡 Normal", "orange"
    else:
        return "🔴 Volatile", "red"


# ---------- history helpers (Part C) ----------
def load_history():
    if not Path(HIST_FILE).exists():
        return []
    out = []
    with open(HIST_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def bar_key(ts_str):
    """Normalize any ISO timestamp to YYYY-MM-DDTHH for dedup / matching."""
    try:
        return pd.to_datetime(ts_str, utc=True).strftime("%Y-%m-%dT%H")
    except Exception:
        return str(ts_str)[:13]


def save_prediction(record, history):
    """Only write if we don't already have a prediction for this target bar."""
    key = bar_key(record["target_ts"])
    if any(bar_key(r.get("target_ts", "")) == key for r in history):
        return  # already saved one for this hour
    with open(HIST_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def fill_actuals(history, df):
    """Back-fill real prices once bars close."""
    if not history:
        return history

    # map normalised hour → close price
    price_map = {row.ts.strftime("%Y-%m-%dT%H"): float(row.close) for row in df.itertuples()}

    updated, changed = [], False
    for rec in history:
        if rec.get("actual") is None:
            k = bar_key(rec.get("target_ts", ""))
            if k in price_map:
                rec = dict(rec)
                rec["actual"]  = price_map[k]
                rec["hit"]     = rec["low95"] <= rec["actual"] <= rec["high95"]
                rec["winkler"] = winkler(rec["low95"], rec["high95"], rec["actual"])
                changed = True
        updated.append(rec)

    if changed:
        with open(HIST_FILE, "w") as f:
            for r in updated:
                f.write(json.dumps(r) + "\n")
    return updated


# ---------- load backtest results (Part A) ----------
@st.cache_data
def load_backtest():
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
    return {
        "coverage": round(np.mean([r["hit"]     for r in rows]), 4),
        "width":    round(np.mean([r["width"]   for r in rows]), 2),
        "winkler":  round(np.mean([r["winkler"] for r in rows]), 2),
        "n":        len(rows),
    }


# ---------- coverage badge ----------
def cov_badge(cov):
    diff = abs(cov - 0.95)
    if diff <= 0.02:
        return "✅ On target"
    elif cov > 0.95:
        return "⚠️ Too wide (over-conservative)" if diff > 0.05 else "⚠️ Slightly wide"
    else:
        return "🔴 Too narrow (overconfident)" if diff > 0.05 else "⚠️ Slightly narrow"


# ---------- charts ----------
def main_chart(df, low, high, current):
    recent    = df.tail(CHART_BARS).copy()
    last_ts   = recent["ts"].iloc[-1]
    next_ts   = last_ts + pd.Timedelta(hours=1)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.02)

    # candles
    fig.add_trace(go.Candlestick(
        x=recent["ts"], open=recent["open"], high=recent["high"],
        low=recent["low"], close=recent["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)

    # shaded forecast band
    fig.add_trace(go.Scatter(
        x=[last_ts, next_ts, next_ts, last_ts],
        y=[current, high, low, current],
        fill="toself", fillcolor="rgba(99,179,237,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="95% band",
    ), row=1, col=1)

    # upper / lower labels
    for price, label in [(high, f"  ${high:,.0f}"), (low, f"  ${low:,.0f}")]:
        fig.add_trace(go.Scatter(
            x=[next_ts], y=[price], mode="markers+text",
            marker=dict(color="#63b3ed", size=10, symbol="line-ew-open", line_width=2),
            text=[label], textposition="middle right",
            textfont=dict(color="#63b3ed", size=11), showlegend=False,
        ), row=1, col=1)

    # dotted current-price line
    fig.add_shape(type="line", x0=last_ts, x1=next_ts, y0=current, y1=current,
                  line=dict(color="#f6c90e", width=1.5, dash="dot"), row=1, col=1)

    # volume
    colours = ["#26a69a" if c >= o else "#ef5350"
               for c, o in zip(recent["close"], recent["open"])]
    fig.add_trace(go.Bar(x=recent["ts"], y=recent["vol"],
                         marker_color=colours, opacity=0.6, showlegend=False), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=11),
        height=500, margin=dict(l=10, r=80, t=30, b=10),
        legend=dict(bgcolor="#1a1d24", bordercolor="#333"),
    )
    for r in [1, 2]:
        fig.update_xaxes(gridcolor="#1e2530", row=r, col=1)
        fig.update_yaxes(gridcolor="#1e2530", row=r, col=1)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def history_chart(history):
    resolved = [r for r in history if r.get("actual") is not None]
    if len(resolved) < 2:
        return None

    dh = pd.DataFrame(resolved)
    dh["prediction_time"] = pd.to_datetime(dh["prediction_time"], utc=True)
    dh = dh.sort_values("prediction_time").tail(96)

    hits   = dh[dh["hit"] == True]
    misses = dh[dh["hit"] == False]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dh["prediction_time"]) + list(dh["prediction_time"])[::-1],
        y=list(dh["high95"]) + list(dh["low95"])[::-1],
        fill="toself", fillcolor="rgba(99,179,237,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="95% band",
    ))
    fig.add_trace(go.Scatter(
        x=dh["prediction_time"], y=dh["actual"],
        mode="lines", name="Actual price", line=dict(color="#f6c90e", width=2),
    ))
    if len(hits):
        fig.add_trace(go.Scatter(
            x=hits["prediction_time"], y=hits["actual"], mode="markers",
            marker=dict(color="#26a69a", size=7), name="Hit ✓",
        ))
    if len(misses):
        fig.add_trace(go.Scatter(
            x=misses["prediction_time"], y=misses["actual"], mode="markers",
            marker=dict(color="#ef5350", size=9, symbol="x"), name="Miss ✗",
        ))
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"), height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor="#1e2530"),
        yaxis=dict(gridcolor="#1e2530", tickprefix="$", tickformat=",.0f"),
    )
    return fig


# ==================== MAIN ====================

# sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    do_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("**Model**")
    st.write(f"- Confidence: 95%")
    st.write(f"- Vol window: {VOL_WINDOW} bars")
    st.write(f"- Simulations: {N_SIM:,}")
    st.write(f"- History: {N_BARS} bars")
    st.markdown("---")
    st.caption("AlphaI × Polaris · GBM + Student-t\nData: data-api.binance.vision")

if do_refresh and HAS_AUTOREFRESH:
    st_autorefresh(interval=60_000, key="refresh")
elif do_refresh:
    st.sidebar.info("pip install streamlit-autorefresh for non-blocking refresh")

# header
st.title("₿  BTC/USDT — Next-Hour 95% Forecaster")
now = datetime.now(timezone.utc)
st.caption(f"Updated: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")

# fetch + predict
with st.spinner("Pulling latest data from Binance..."):
    try:
        df  = get_btc_data()
        low, high, current, vol_ann = predict(df["close"].values)
        err = None
    except Exception as e:
        err = str(e)

if err:
    st.error(f"Something went wrong: {err}")
    st.stop()

next_bar = df["ts"].iloc[-1] + pd.Timedelta(hours=1)
secs_left = max(0, (next_bar.to_pydatetime() - now).total_seconds())
mins_left = int(secs_left // 60)
secs_left = int(secs_left % 60)

# ---------- Part A metrics ----------
bt = load_backtest()
st.markdown("### 📊 Backtest metrics (30-day, Part A)")

if bt:
    a, b, c, d = st.columns(4)
    a.metric("Coverage @95%", f"{bt['coverage']:.4f}",
             delta=f"{bt['coverage']-0.95:+.4f} vs target", delta_color="normal",
             help="Target is ~0.95. Higher = too wide, lower = too narrow.")
    b.metric("Avg width", f"${bt['width']:,.0f}",
             help="Average size of predicted range. Narrower is better if coverage holds.")
    c.metric("Winkler ↓", f"${bt['winkler']:,.0f}",
             help="Combined accuracy+tightness score. Lower is better.")
    d.metric("Predictions", f"{bt['n']:,}")
    st.caption(f"Coverage verdict: **{cov_badge(bt['coverage'])}**")
else:
    st.info("No backtest_results.jsonl found. Run btc_gbm_backtest.py and push the file.")

st.divider()

# ---------- live prediction ----------
reg_label, reg_color = regime(vol_ann)

left, right = st.columns([3, 1])
with left:
    st.markdown("### 🔮 Next-hour prediction")
with right:
    st.markdown(f"**Vol regime:** {reg_label} ({vol_ann:.1f}% ann.)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC price now", f"${current:,.2f}")
c2.metric("Lower bound (2.5%)", f"${low:,.2f}", delta=f"{(low/current-1)*100:+.2f}%")
c3.metric("Upper bound (97.5%)", f"${high:,.2f}", delta=f"{(high/current-1)*100:+.2f}%")
c4.metric("Range width", f"${high-low:,.0f}", delta=f"{(high-low)/current*100:.2f}%")

st.markdown(
    f"⏱ **{mins_left}m {secs_left:02d}s** until bar closes · "
    f"Target: `{next_bar.strftime('%Y-%m-%d %H:00 UTC')}` · "
    f"Range: **${low:,.0f} – ${high:,.0f}**"
)

fig = main_chart(df, low, high, current)
st.plotly_chart(fig, use_container_width=True)

# ---------- Part C: save + show history ----------
record = {
    "prediction_time": now.isoformat(),
    "current_ts":      df["ts"].iloc[-1].isoformat(),
    "target_ts":       next_bar.isoformat(),
    "price":           current,
    "low95":           low,
    "high95":          high,
    "width":           high - low,
    "actual":          None,
    "hit":             None,
    "winkler":         None,
}

history = load_history()
save_prediction(record, history)
history = load_history()
history = fill_actuals(history, df)

st.divider()

resolved = [r for r in history if r.get("actual") is not None]
pending  = [r for r in history if r.get("actual") is None]

st.markdown(f"### 📈 Prediction history — {len(resolved)} resolved / {len(pending)} pending")

if resolved:
    hits      = [r for r in resolved if r.get("hit")]
    live_cov  = len(hits) / len(resolved)
    live_wink = float(np.mean([r["winkler"] for r in resolved]))
    live_w    = float(np.mean([r["width"]   for r in resolved]))

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Live coverage",  f"{live_cov:.4f}")
    h2.metric("Live Winkler ↓", f"${live_wink:,.0f}")
    h3.metric("Avg width",      f"${live_w:,.0f}")
    h4.metric("Resolved",       len(resolved))
    st.caption(f"Live verdict: **{cov_badge(live_cov)}**")

    fig_h = history_chart(history)
    if fig_h:
        st.plotly_chart(fig_h, use_container_width=True)
else:
    st.info("No resolved predictions yet — check back after the next bar closes (~1 hour).")

with st.expander("Raw prediction log"):
    if history:
        dh = pd.DataFrame(history[::-1])
        cols = [c for c in ["target_ts", "price", "low95", "high95", "width", "actual", "hit", "winkler"] if c in dh.columns]
        display = dh[cols].copy()
        for col in ["price", "low95", "high95", "width", "actual", "winkler"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) and x is not None else "—")
        display["hit"] = display["hit"].apply(lambda x: "✅" if x is True else ("❌" if x is False else "⏳"))
        st.dataframe(display, use_container_width=True, height=280)
    else:
        st.write("Nothing yet.")

st.divider()
st.caption("AlphaI × Polaris Challenge · GBM + Student-t · Binance public API · No API key needed")