<div align="center">

# ₿ BTC/USDT Next-Hour Forecaster

### AlphaI × Polaris Challenge — Probabilistic Bitcoin Range Forecasting

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Predict the next hour's BTC/USDT price range with a 95% confidence interval, not just a single number.**

[Quick Start](#-quick-start) • [Repo Layout](#-repository-structure) • [Files](#-what-each-file-does) • [How It Works](#-how-it-works)

---

</div>

## 📊 What this repo does

This repo contains a complete pipeline for next-hour BTC/USDT range forecasting using a GBM model with Student-t noise.

It includes:

- a **Colab notebook** for exploration and backtesting,
- a **Python backtest script** to regenerate results locally,
- a **Streamlit dashboard** to show live predictions and history,
- and a **live history log** that stores resolved predictions over time.

This is ideal for anyone who wants to evaluate probabilistic crypto forecasts, track live performance, and share a working dashboard.

---

## 📁 Repository structure

```
btc-gbm-forecaster/
├── 📓 Copy of GBM.ipynb              # Backtest notebook for Part A
├── 🐍 btc_gbm_backtest.py            # Standalone script to regenerate backtest_results.jsonl
├── 🎨 dashboard.py                   # Streamlit dashboard for live predictions and history
├── 📦 requirements.txt               # Python dependencies
├── 📊 backtest_results.jsonl         # Backtest output (commit this file after running)
├── 📈 prediction_history.jsonl       # Live prediction log updated by the dashboard
├── 📖 README.md                      # This file
```

> Note: this repo does not contain a `docs/` folder.

---

## ⚡ Quick start

### 1. Run the dashboard locally

```bash
git clone https://github.com/aneek22112007-tech/btc-gbm-forecaster.git
cd btc-gbm-forecaster
pip install -r requirements.txt
streamlit run dashboard.py
```

Then open **http://localhost:8501** in your browser.

### 2. Regenerate backtest results

```bash
python btc_gbm_backtest.py
```

This downloads BTCUSDT hourly candles, runs the walk-forward backtest, and saves `backtest_results.jsonl`.

### 3. Use the notebook

Open `Copy of GBM.ipynb` in Google Colab and run the cells there if you want a notebook workflow.

---

## 💡 What each file does

- `Copy of GBM.ipynb` — interactive Colab notebook version of the backtest
- `btc_gbm_backtest.py` — Python script that saves `backtest_results.jsonl`
- `dashboard.py` — Streamlit app for live predictions and history
- `requirements.txt` — Python package list
- `backtest_results.jsonl` — backtest results used by the dashboard
- `prediction_history.jsonl` — historical prediction log used by the dashboard
- `README.md` — this guide

---

## 🚀 Why this approach

This project is not just about predicting a single price. It predicts a **95% confidence range**, which is more useful for decisions when markets are volatile.

The model is designed to be:

- **honest** about uncertainty,
- **responsive** to recent volatility,
- **robust** to fat-tailed returns.

---

## 🧠 How it works

### Backtest flow

- Fetch recent BTCUSDT hourly bars
- Run a rolling walk-forward simulation
- Fit volatility and Student-t parameters on recent returns
- Simulate future returns
- Compute the 95% prediction range
- Evaluate coverage, width, and Winkler score

### Dashboard flow

`dashboard.py` shows:

- current BTC price
- next-hour low/high range
- range width
- backtest health metrics
- prediction history with hit/miss indicators

It also appends predictions to `prediction_history.jsonl` so history builds over time.

---

## 📝 Notes

- Keep `backtest_results.jsonl` committed if you want dashboard metrics to reflect the latest backtest.
- `prediction_history.jsonl` is updated whenever the dashboard runs.
- The dashboard will still start without `backtest_results.jsonl`, but the historical metrics section will be empty until the file exists.

---

## 🛠️ Dependencies

Install packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 📌 Quick commands

```bash
streamlit run dashboard.py
python btc_gbm_backtest.py
```

---

## 📬 Feedback

If you want to improve this repo, the best next step is:

1. regenerate `backtest_results.jsonl`
2. run the dashboard
3. inspect `prediction_history.jsonl`

Thanks for checking out the BTC GBM Forecaster.
