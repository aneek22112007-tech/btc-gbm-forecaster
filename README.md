<div align="center">

# ₿ BTC/USDT Next-Hour Forecaster

### AlphaI × Polaris Challenge — Advanced Probabilistic Forecasting

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](https://github.com)

**Predict Bitcoin's next-hour 95% confidence interval using Geometric Brownian Motion + Student-t simulation**

[Live Demo](#-live-dashboard) • [Features](#-key-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation)

---

</div>

## 📊 Overview

This project implements a **probabilistic forecasting system** for Bitcoin (BTC/USDT) that predicts the next hour's price range with 95% confidence. Unlike point predictions, this approach provides **uncertainty quantification** — critical for risk management in volatile crypto markets.

### 🎯 What Makes This Special

- **🔬 Rigorous Backtesting** — 30-day walk-forward validation with strict no-peeking
- **📈 Live Dashboard** — Real-time predictions with auto-refresh and historical tracking
- **🎲 Advanced Statistics** — Student-t distribution for fat-tailed returns
- **⚡ Volatility Clustering** — EWMA-based adaptive volatility estimation
- **🎨 Beautiful UI** — Interactive charts with volume bars and prediction bands
- **🔄 Auto-Refresh** — Non-blocking 60-second updates
- **📊 Performance Metrics** — Coverage, Winkler score, and health indicators

---

## 🚀 Key Features

### Part A: Backtesting Engine
- ✅ **Walk-forward validation** on 720+ hours of data
- ✅ **Strict no-peeking** — structurally impossible to leak future data
- ✅ **EWMA volatility** (λ=0.94 RiskMetrics) for clustering
- ✅ **Adaptive Student-t** — MLE fit on sliding window
- ✅ **Itô drift correction** for unbiased GBM
- ✅ **15,000 Monte Carlo simulations** per prediction
- ✅ **Regime detection** (calm/normal/volatile)

### Part B: Live Dashboard
- ✅ **Real-time predictions** from Binance public API (no key needed)
- ✅ **Interactive charts** with Plotly (candlesticks + volume)
- ✅ **Auto-refresh** every 60 seconds (non-blocking)
- ✅ **Countdown timer** to next bar close
- ✅ **Volatility regime indicator** (🟢 calm / 🟡 normal / 🔴 volatile)
- ✅ **Coverage health gauge** (green/yellow/red)

### Part C: Prediction History
- ✅ **Persistent tracking** via session state + JSONL backup
- ✅ **Automatic back-filling** of actual prices
- ✅ **Hit/miss visualization** with color-coded markers
- ✅ **Live coverage metrics** updated in real-time
- ✅ **Deduplication** — one prediction per hour
- ✅ **Historical chart** showing last 96 predictions

---

## 📁 Repository Structure

```
btc-gbm-forecaster/
├── 📓 Copy of GBM.ipynb              # Colab notebook (Part A)
├── 🐍 btc_gbm_backtest.py            # Standalone backtest script
├── 🎨 dashboard.py                   # Streamlit dashboard (Parts B + C)
├── 📦 requirements.txt               # Python dependencies
├── 📊 backtest_results.jsonl         # Backtest output (commit after running)
├── 📈 prediction_history.jsonl       # Live prediction log
├── 📖 README.md                      # This file
└── 📚 docs/
    ├── BUG_FIXES_SUMMARY.md          # Detailed bug fix documentation
    ├── CHANGES_QUICK_REFERENCE.md    # Quick reference guide
    ├── BEFORE_AFTER_EXAMPLES.md      # Code comparison examples
    └── DEPLOYMENT_READY.md           # Deployment checklist
```

---

## ⚡ Quick Start

### Option 1: Run Locally (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/btc-gbm-forecaster.git
cd btc-gbm-forecaster

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
```

Open your browser to **http://localhost:8501** 🎉

### Option 2: Deploy to Streamlit Cloud (Free!)

1. **Push to GitHub** (public or private repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repo, branch `main`, file `dashboard.py`
5. Click **"Deploy"**
6. Get your public URL: `https://yourname-btc-forecaster.streamlit.app`

> **Note:** Streamlit Cloud apps "sleep" after 7 days of inactivity but wake up in ~30 seconds when visited.

### Option 3: Run Backtest

```bash
# Run 30-day backtest (generates backtest_results.jsonl)
python btc_gbm_backtest.py

# Or use the Colab notebook
# Open Copy of GBM.ipynb in Google Colab
# Runtime → Run all
```

---

## 🎨 Live Dashboard

### Main Features

<table>
<tr>
<td width="50%">

#### 📊 Real-Time Prediction
- Current BTC price
- 95% confidence interval
- Lower & upper bounds
- Range width
- Countdown to next bar

</td>
<td width="50%">

#### 📈 Interactive Charts
- Candlestick price chart
- Volume bars (color-coded)
- Prediction band overlay
- Historical hit/miss markers
- 96-hour lookback

</td>
</tr>
<tr>
<td width="50%">

#### 🎯 Performance Metrics
- Backtest coverage (30-day)
- Average interval width
- Winkler score
- Live coverage tracking
- Health indicators

</td>
<td width="50%">

#### ⚙️ Smart Features
- Auto-refresh (60s)
- Session state persistence
- Duplicate prevention
- Actual price back-filling
- Data quality warnings

</td>
</tr>
</table>

---

## 🔬 Technical Deep Dive

### 1. No-Peeking Guarantee

The backtest uses **strict temporal ordering** to prevent data leakage:

```python
for i in range(warmup, len(closes) - 1):
    history = closes[:i+1]   # Only data up to time i
    actual  = closes[i+1]    # Future data (revealed AFTER prediction)
    
    pred = fit_and_predict(history)  # Model sees only history
```

The slice `closes[:i+1]` makes it **structurally impossible** to leak future information.

### 2. Volatility Clustering

Bitcoin exhibits **volatility clustering** — high volatility periods cluster together. We capture this with EWMA:

```python
def ewma_vol(log_returns, lam=0.94):
    """EWMA (RiskMetrics) volatility"""
    var = initial_variance
    for r in log_returns:
        var = lam * var + (1 - lam) * r**2
    return sqrt(var)
```

**Why EWMA?**
- Reacts faster to regime changes than rolling window
- Gives more weight to recent observations
- λ=0.94 is the RiskMetrics standard

### 3. Fat-Tailed Returns (Student-t)

Bitcoin returns have **fat tails** (ν ≈ 3-5), meaning extreme moves are more common than a Gaussian would predict:

```python
# Fit Student-t distribution
nu, mu, scale = student_t.fit(log_returns, floc=0)

# Simulate with fat tails
sim_returns = student_t.rvs(df=nu, loc=mu, scale=vol, size=15000)
sim_prices = S0 * np.exp(sim_returns)

# Extract 95% confidence interval
low  = np.percentile(sim_prices, 2.5)
high = np.percentile(sim_prices, 97.5)
```

**Why Student-t?**
- Captures extreme events better than Gaussian
- Empirically validated for crypto returns
- Adaptive ν parameter via MLE

### 4. Itô Drift Correction

For unbiased GBM simulation, we apply the **Itô correction**:

```python
mu = mean(log_returns) - 0.5 * vol**2  # Drift correction
```

This ensures the expected value of the simulated price equals the forward price.

---

## 📊 Performance Metrics

### Coverage (@95%)
**Target:** ≈ 0.95 (95% of actuals should fall within predicted range)

- **0.93 - 0.97** → 🟢 Healthy (on target)
- **0.90 - 0.93 or 0.97 - 1.00** → 🟡 Acceptable (slightly off)
- **< 0.90 or > 1.00** → 🔴 Poor (needs tuning)

### Average Width
**Target:** As narrow as possible while maintaining coverage ≈ 0.95

- Narrower intervals = more precise predictions
- But too narrow → coverage drops below 0.95
- Trade-off between precision and reliability

### Winkler Score
**Target:** Lower is better

Combines **accuracy** (coverage) and **precision** (width) into a single metric:

```python
def winkler_score(low, high, actual, alpha=0.05):
    width = high - low
    if actual < low:
        return width + (2/alpha) * (low - actual)  # Penalty for miss
    elif actual > high:
        return width + (2/alpha) * (actual - high)  # Penalty for miss
    return width  # No penalty if hit
```

- **Lower score** = better model
- Penalizes misses heavily (2/α = 40× multiplier)
- Rewards tight intervals when they hit

---

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.8+** — Core language
- **NumPy** — Numerical computing
- **Pandas** — Data manipulation
- **SciPy** — Statistical distributions
- **Streamlit** — Web dashboard framework
- **Plotly** — Interactive charts
- **Requests** — API calls to Binance

### Data Source
- **Binance Public API** (`data-api.binance.vision`)
- No API key required
- No geo-blocking (works in India)
- 1-hour BTCUSDT candlesticks
- Historical data + real-time updates

### Architecture
```
┌─────────────────┐
│  Binance API    │  ← Fetch BTCUSDT 1h bars
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Pipeline  │  ← Clean, validate, transform
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GBM + Student-t│  ← Fit model, simulate paths
│  Prediction     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit UI   │  ← Display, track, visualize
└─────────────────┘
```

---

## 🎯 Model Configuration

### Constants (Tunable)

```python
N_BARS        = 500    # Historical bars for model fitting
CHART_BARS    = 50     # Bars displayed on chart
VOL_WINDOW    = 20     # Volatility estimation window
LOOKBACK      = 100    # Student-t fitting window
N_SIM         = 10000  # Monte Carlo simulations
ALPHA         = 0.05   # Confidence level (95% = 1 - 0.05)
```

### Volatility Regimes

| Regime | Annualized Vol | Indicator | Interpretation |
|--------|----------------|-----------|----------------|
| **Calm** | < 40% | 🟢 Green | Low volatility, tight ranges |
| **Normal** | 40% - 80% | 🟡 Yellow | Typical BTC volatility |
| **Volatile** | > 80% | 🔴 Red | High volatility, wide ranges |

---

## 🐛 Bug Fixes & Improvements

This version includes **8 critical bug fixes** and **7 dashboard improvements**:

### Fixed Bugs ✅
1. **Timestamp matching** — Actuals now back-fill correctly
2. **Duplicate predictions** — One prediction per hour
3. **History persistence** — Session state + file backup
4. **Raw table display** — Shows actual column properly
5. **Lookback consistency** — Matches backtest parameters
6. **Auto-refresh** — Non-blocking with st_autorefresh
7. **Winkler key** — Verified working correctly
8. **Data quality** — Warns on insufficient data

### Dashboard Improvements ✅
1. Coverage health indicator (green/yellow/red)
2. Volatility regime indicator (enhanced)
3. Auto-refresh with st_autorefresh
4. Next bar countdown (verified)
5. Volume bars on chart (verified)
6. Better raw log table formatting
7. Data quality warnings

See [BUG_FIXES_SUMMARY.md](docs/BUG_FIXES_SUMMARY.md) for detailed documentation.

---

## 📚 Documentation

### Core Documentation
- **[README.md](README.md)** — This file (overview & quick start)
- **[BUG_FIXES_SUMMARY.md](docs/BUG_FIXES_SUMMARY.md)** — Detailed bug fix documentation
- **[CHANGES_QUICK_REFERENCE.md](docs/CHANGES_QUICK_REFERENCE.md)** — Quick reference guide
- **[BEFORE_AFTER_EXAMPLES.md](docs/BEFORE_AFTER_EXAMPLES.md)** — Code comparison examples
- **[DEPLOYMENT_READY.md](docs/DEPLOYMENT_READY.md)** — Deployment checklist

### Code Documentation
All functions include docstrings with:
- Purpose and behavior
- Parameters and return values
- Implementation notes
- Bug fix references (where applicable)

---

## 🚀 Deployment Options

### Streamlit Cloud (Recommended)
- ✅ **Free tier available**
- ✅ **Automatic HTTPS**
- ✅ **GitHub integration**
- ✅ **Auto-deploy on push**
- ⚠️ Sleeps after 7 days (wakes in ~30s)

### HuggingFace Spaces
```bash
# Create a new Space (Streamlit)
# Upload files: dashboard.py, requirements.txt, backtest_results.jsonl
# Auto-deploys on commit
```

### Railway / Render
```bash
# Add Procfile:
web: streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0
```

### Google Cloud Run
```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD streamlit run dashboard.py --server.port 8080 --server.address 0.0.0.0
```

### Docker (Local)
```bash
docker build -t btc-forecaster .
docker run -p 8501:8501 btc-forecaster
```

---

## 🧪 Testing & Validation

### Run Tests
```bash
# Syntax check
python -m py_compile dashboard.py
python -m py_compile btc_gbm_backtest.py

# Run backtest
python btc_gbm_backtest.py

# Run dashboard locally
streamlit run dashboard.py
```

### Validation Checklist
- ✅ No syntax errors
- ✅ No linting warnings
- ✅ Backtest coverage ≈ 0.95
- ✅ Dashboard loads without errors
- ✅ Auto-refresh works
- ✅ Predictions save correctly
- ✅ Actuals back-fill after 1 hour
- ✅ No duplicate predictions

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write descriptive commit messages

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **AlphaI × Polaris** — Challenge organizers
- **Binance** — Public API for historical data
- **Streamlit** — Amazing dashboard framework
- **SciPy** — Statistical distributions
- **Plotly** — Interactive charting

---

## 📞 Support & Contact

### Issues & Questions
- 🐛 **Bug reports:** [Open an issue](https://github.com/yourusername/btc-gbm-forecaster/issues)
- 💡 **Feature requests:** [Open an issue](https://github.com/yourusername/btc-gbm-forecaster/issues)
- 📧 **Email:** your.email@example.com

### Resources
- 📖 [Streamlit Documentation](https://docs.streamlit.io/)
- 📊 [Binance API Docs](https://binance-docs.github.io/apidocs/)
- 🎓 [Student-t Distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
- 📈 [Geometric Brownian Motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)

---

## 🎯 FAQ

<details>
<summary><b>Q: Why Student-t instead of Gaussian?</b></summary>

Bitcoin returns have **fat tails** (kurtosis > 3), meaning extreme moves happen more often than a Gaussian would predict. Student-t with ν ≈ 3-5 captures this behavior, leading to better coverage during volatile periods.
</details>

<details>
<summary><b>Q: How does the model handle volatility clustering?</b></summary>

We use **EWMA (Exponentially Weighted Moving Average)** with λ=0.94, which gives more weight to recent observations. This allows the model to quickly adapt to regime changes (calm → volatile or vice versa).
</details>

<details>
<summary><b>Q: What if Binance is geo-blocked in my region?</b></summary>

We use `data-api.binance.vision`, the **public mirror** that bypasses geo-restrictions. It works in India and most other regions. No API key required!
</details>

<details>
<summary><b>Q: Can I use a different model?</b></summary>

Yes! The codebase is modular. Replace the `predict()` function in `dashboard.py` with your own model. Just ensure it returns `(low, high, current, vol_ann)` in the same format.
</details>

<details>
<summary><b>Q: How do I improve coverage if it's too low?</b></summary>

- Increase `N_SIM` (more simulations → smoother quantiles)
- Increase `VOL_WINDOW` (wider volatility estimate)
- Adjust `ALPHA` (e.g., 0.03 for 97% confidence)
- Use a lower `nu` value (fatter tails)
</details>

<details>
<summary><b>Q: Why does the dashboard show "⏳" for some predictions?</b></summary>

The "⏳" symbol means the prediction is **pending** — the target hour hasn't closed yet, so we don't have the actual price. After the hour closes, the dashboard automatically back-fills the actual price and shows ✅ (hit) or ❌ (miss).
</details>

<details>
<summary><b>Q: How long does the backtest take?</b></summary>

On a typical laptop, the backtest runs in **2-5 minutes** for 720 predictions (30 days). On Google Colab, it's usually **1-2 minutes**.
</details>

<details>
<summary><b>Q: Can I deploy this for free?</b></summary>

Yes! **Streamlit Cloud** offers a free tier that's perfect for this project. You get:
- Free hosting
- Automatic HTTPS
- GitHub integration
- Auto-deploy on push

The only limitation is that apps sleep after 7 days of inactivity (but wake up in ~30 seconds when visited).
</details>

---

<div align="center">

## ⭐ Star This Repo!

If you find this project useful, please consider giving it a star ⭐

**Made with ❤️ for the AlphaI × Polaris Challenge**

[⬆ Back to Top](#-btcusdt-next-hour-forecaster)

</div>
