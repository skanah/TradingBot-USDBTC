# TradingBot-USDBTC 🤖📈

An adaptive, self-optimizing **BTCUSD scalping bot** for **MetaTrader 5** that combines a robust Genetic Algorithm (GA), online learning, ML-based signal filtering, and strict capital-protection controls — all in pure Python.

> ⚠️ **DISCLAIMER:** This software trades **real money** on live broker accounts. Trading cryptocurrencies is extremely risky. This project is provided for educational and research purposes **without any warranty**. Past backtest or live performance does not guarantee future results. Use at your own risk — you are solely responsible for any financial losses.

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Architecture](#-architecture)
4. [Repository Structure](#-repository-structure)
5. [Module Details](#-module-details)
6. [Safety & Risk Controls](#-safety--risk-controls)
7. [Prerequisites](#-prerequisites)
8. [Installation](#-installation)
9. [MetaTrader 5 Setup](#-metatrader-5-setup)
10. [Usage](#-usage)
11. [Configuration (production_loop.py)](#-configuration-production_looppy)
12. [Data Sources](#-data-sources)
13. [Models & Logs](#-models--logs)
14. [Roadmap Ideas](#-roadmap-ideas)

---

## 🔭 Overview

`TradingBot-USDBTC` is a production-grade scalping system for the **BTCUSD** symbol. It:

- Polls live BTCUSD indicator/candle data (with fallback to direct MT5 prices).
- **Evolves its own trading parameters** using a robust Genetic Algorithm with Walk-Forward Optimization (WFO), so strategies are validated out-of-sample instead of curve-fitted.
- **Learns continuously** from live trade outcomes and switches between operational modes (Reconnaissance / Scalping / Aggressive / Defensive).
- Optionally blends in a **linear ML model** and an external **API signal filter** for dual-confirmation entries.
- Executes **real 0.01-lot trades** on MetaTrader 5 via the official `MetaTrader5` Python package, with breakeven protection and software trailing stops.
- Enforces hard capital-protection limits (daily loss cap, consecutive-loss lockout, minimum-balance halt, spread filter).

The project prioritizes **real-world profitability over backtest fantasy**: realistic transaction costs (spread, commission, slippage, friction buffer), regime-aware stress testing (COVID crash 2020, 2022 bear market, consolidation periods), and bootstrap confidence intervals are all baked into the fitness function.

---

## ✨ Key Features

| Area | What it does |
|---|---|
| 🧬 Robust GA optimization | Walk-forward optimization (70/30 rolling), multi-period cross-validation, regime stress tests, parameter sensitivity bounds, bootstrap confidence intervals |
| 📊 Realistic cost modeling | Spread, commission (~$0.17 round-trip), regime-dependent slippage, and a friction buffer — calibrated from live broker logs for 0.01-lot BTCUSD economics |
| 🧠 Online learning engine | Tracks up to 100 recent trades, computes win rate / profit factor / hold-time stats, auto-switches between 4 operating modes |
| 🤖 ML signal filter | Optional linear model (`models/ml_model.json`) trained from history with L2 regularization; signals only fire when ML and GA agree |
| 🖥️ Tkinter GUI | Live dashboard: balance, equity, open positions, recent trades, mode indicator — connects directly to MT5 if the bot isn't running |
| 🛡️ Capital protection | Max daily loss, max consecutive losses, minimum balance halt, spread filter, cooldown between trades, breakeven + trailing stop management |
| 🗂️ Model versioning | Automatic archiving of every improved parameter set, fitness tracking, and rollback support |
| 🧪 Dry-run first | `production_loop.py` defaults to `DRY_RUN=1` — no trade is sent unless you explicitly configure an exchange adapter |

---

## 🏗 Architecture

```
                    ┌──────────────────────────────┐
   obilo.io feeds ─▶│  robust_ga_framework.py      │
   (live + history) │  RobustGA + WFO + Backtester │
                    └──────────┬───────────────────┘
                               │ best params (fitness-gated)
                               ▼
                    ┌──────────────────────────────┐
                    │  model_versioning.py         │
                    │  save / archive / rollback   │
                    └──────────┬───────────────────┘
                               │ models/robust_params.json
                               ▼
      ┌────────────────────────┴───────────────────────────┐
      │                  scalping_bot_v3.py                 │
      │  LiveBot ── MT5Connection (real 0.01-lot orders)   │
      │  + online_learning.py (adaptive mode switching)     │
      │  + trading_gui.py (Tkinter dashboard)               │
      └─────────────────────────────────────────────────────┘
                               ▲
                    ┌──────────┴───────────────────┐
                    │  production_loop.py          │
                    │  5-min polling loop, ML      │
                    │  filter, dry_run/http/mt5    │
                    │  adapters, CSV signal log    │
                    └──────────────────────────────┘
```

**Entry strategy (dual confirmation):** a trade is only opened when short-period momentum, medium-period momentum, and (optionally) the external API signal all agree in the same direction, subject to GA-evolved thresholds and ML score checks.
---

## 📁 Repository Structure

```
TradingBot-USDBTC/
├── scalping_bot_v3.py        # Main live-trading bot (MT5, GA re-evolution, GUI, learning)
├── robust_ga_framework.py     # Robust GA: WFO, backtester, regime stress tests
├── production_loop.py        # Safe production polling loop (dry-run by default)
├── online_learning.py        # Adaptive learning engine + operating modes
├── model_versioning.py       # Model save/archive/rollback + fitness tracker
├── trading_gui.py            # Tkinter real-time trading dashboard
├── test_mt5_connection.py    # Quick MT5 connectivity test
├── qwen/
│   └── settings.json         # Local Ollama (qwen-3b) LLM endpoint settings
└── models/                   # Evolved parameters, metadata, and version archive
```

Runtime artifacts (logs, CSV histories, `venv/`, `__pycache__/`) are generated locally and excluded via `.gitignore`.

---

## 📦 Module Details

### `scalping_bot_v3.py` — Main Live Bot
The primary entry point. Connects to MetaTrader 5 on the same Windows PC, opens **real 0.01-lot BTCUSD trades**, and manages them with:
- Dual-confirmation momentum entries (short + medium momentum, plus optional API filter)
- Breakeven move at **$0.60 floating profit** (covers commission + swap)
- GA-evolved software trailing stop (activation/distance optimized per generation)
- Periodic **re-evolution** with `RobustGA` every 30 minutes (requires ≥ 1,000 candles)
- Graceful Ctrl+C shutdown that reports open positions and final balance

### `robust_ga_framework.py` — Robust Genetic Algorithm
Anti-overfitting optimization engine:
- **Walk-Forward Optimization** — rolling 70% in-sample / 30% out-of-sample validation
- **Multi-period cross-validation** across market regimes (Trending Up/Down, Ranging, Volatile, Calm)
- **Robustness metrics** — Sharpe ratio, max drawdown, Calmar ratio, profit factor, win rate, exposure ratio
- **Bootstrap confidence intervals** for statistical significance of results
- **Transaction cost sensitivity analysis** — how quickly profits evaporate as costs increase
- **Historical regime stress tests** — COVID-19 crash (2020), 2022 bear market, 2021/2023 consolidations
- **Hard fitness cap** (`MAX_FITNESS_CAP`) that rejects fantasy fitness scores from degenerate/overfit parameter sets



### `production_loop.py` — Safe Production Loop
A standalone, safety-first signal engine:
- Polls the live data URL every 5 minutes and ingests only the newest tail candle
- Runs each candle through saved **ML coefficients** (`models/ml_model.json`) and GA thresholds
- Sends the resulting action to a pluggable adapter: `dry_run` (default), `http`, or `mt5`
- Logs every decision to `production_signals.csv` with full reasons and confidence
- Defaults to **DRY_RUN** — no trades unless explicitly enabled

### `online_learning.py` — Online Learning Engine
- Maintains a rolling window of the last **100 trade outcomes**
- Periodically computes win rate, average win/loss, profit factor, hold time, and best-performing direction/momentum setup
- Automatically switches operating modes:

| Mode | Behavior |
|---|---|
| `RECONNAISSANCE` | Learning mode — more trades, lower thresholds, longer holds |
| `SCALPING` | Normal trading (default) |
| `AGGRESSIVE` | Faster profit taking after strong performance |
| `DEFENSIVE` | Protection mode after losses — 2× cooldown, higher thresholds |

### `model_versioning.py` — Model Versioning System
- Saves a new version whenever fitness improves by > 0.05 over the current best
- Archives up to **20 previous versions** in `models/archive/`
- Tracks the all-time best fitness and full improvement history
- Always loads the highest-fitness validated model at startup

### `trading_gui.py` — Tkinter Dashboard
A standalone real-time dashboard showing account balance, equity, open positions, and recent trade history. It can attach to a running `LiveBot` instance or connect directly to MT5 (with retry logic).

### `test_mt5_connection.py`
Minimal script to verify that the `MetaTrader5` package can reach your terminal and read account info (login, balance, equity). Run this first when setting up.

---

## 🛡️ Safety & Risk Controls

Hard-coded in `scalping_bot_v3.py` for a small (~$134) live account; adjust to your own capital:

| Control | Default | Effect |
|---|---|---|
| `MAX_DAILY_LOSS` | `-5.0` | Halt trading after $5 daily loss (~3.7% of account) |
| `MAX_CONSECUTIVE_LOSSES` | `3` | Lockout after 3 losses in a row |
| `MIN_ACCOUNT_BALANCE` | `50.0` | Absolute floor — halt trading below this balance |
| `MAX_SPREAD` | `100.0` | Skip entries when broker spread is too wide |
| `COOLDOWN_MIN` | `120` s | Minimum time between trades |
| `BREAKEVEN_PROFIT` | `$0.60` | Move SL to entry to lock in net profit after fees |
| Trailing stop | GA-evolved | Software trailing stop activates/locks profit as price moves |
| `MAX_FITNESS_CAP` | `250`+ | Rejects implausible (overfit) fitness scores |

Live parameters loaded from saved GA models are additionally clamped to safety bounds: take-profit ∈ [80, 250], stop-loss ∈ [100, 250], cooldown ≥ 60 s.


---

## 🔧 Prerequisites

- **Windows** (required by the `MetaTrader5` Python package)
- **Python 3.10+** (developed and tested on 3.13)
- **MetaTrader 5** terminal installed, running, and logged in
- Python packages: `MetaTrader5`, `numpy`, `requests` (Tkinter ships with Python)

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/skanah/TradingBot-USDBTC.git
cd TradingBot-USDBTC

# 2. (Optional but recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install MetaTrader5 numpy requests
```

## 🖥️ MetaTrader 5 Setup

1. Open MetaTrader 5 and **log in** to your (demo!) account.
2. Go to **Tools → Options → Expert Advisors**.
3. Enable **"Allow Automated Trading"**.
4. Make sure **BTCUSD** is visible in **Market Watch** (check the exact symbol name your broker uses — e.g. `BTCUSD.a`, `BTCUSDm` — and update `MT5_SYMBOL` if needed).
5. Verify connectivity: `python test_mt5_connection.py`

## 🎮 Usage

### 1. Test the MT5 connection
```bash
python test_mt5_connection.py
```

### 2. Evolve trading parameters (Robust GA)
```bash
python robust_ga_framework.py
```
Fetches historical data, runs walk-forward-optimized evolution, and saves the best validated parameter set to `models/robust_params.json`.

### 3. Run the live scalping bot
```bash
python scalping_bot_v3.py
```
Opens real 0.01-lot trades on your MT5 account. Press **Ctrl+C** for a graceful shutdown. The Tkinter GUI launches automatically when available.

### 4. Run the GUI standalone
```bash
python trading_gui.py
```

### 5. Production signal loop (safe by default)
```bash
# Dry run — logs signals to production_signals.csv, sends nothing
python production_loop.py --once

# Train the ML model from historical data, then run
python production_loop.py --train-ml --ml-epochs 1000 --ml-lr 0.1 --ml-l2 0.001

# Continuous polling with the MT5 adapter (real orders!)
python production_loop.py --adapter mt5
```
---

## ⚙️ Configuration (`production_loop.py`)

All values can be set via environment variables or CLI flags:

| Env var | Default | Description |
|---|---|---|
| `LIVE_DATA_URL` | `https://obilo.io/casdata/BTCUSDlive.txt` | Live indicator feed |
| `HIST_DATA_URL` | `https://obilo.io/casdata/BTCUSD.txt` | Historical candles |
| `POLL_INTERVAL_SECONDS` | `300` | Seconds between polls |
| `LIVE_TAIL_CANDLES` | `120` | Candles kept in the rolling window |
| `LIVE_MAX_DATA_AGE_SECONDS` | `900` | Reject stale live data |
| `SYMBOL` | `BTCUSD` | Traded symbol |
| `EXCHANGE_ADAPTER` | `dry_run` | `dry_run`, `http`, or `mt5` |
| `EXCHANGE_API_URL` / `EXCHANGE_API_TOKEN` | *(empty)* | HTTP adapter credentials |
| `DRY_RUN` | `1` (unless `EXCHANGE_API_URL` set) | Set `0` to allow real trades |

Key CLI flags: `--adapter`, `--interval`, `--symbol`, `--dry-run`, `--once`, `--train-ml`, `--ml-epochs`, `--ml-lr`, `--ml-l2`, `--send-hold-signals`.
---

## 🌐 Data Sources

- **Live feed** — `https://obilo.io/casdata/BTCUSDlive.txt`: comma-separated live indicator values (momentum short/medium, range, close position, trend, API signal score, etc.)
- **Historical candles** — `https://obilo.io/casdata/BTCUSD.txt`: CSV rows of timestamped OHLCV candles used for backtesting, WFO, and ML training
- **MT5 prices** — used directly for entries, exits, spreads, and position management when connected to a live terminal

---

## 🗂️ Models & Logs

| Path | Purpose |
|---|---|
| `models/robust_params.json` | Current best RobustGA parameter set |
| `models/best_params.json` | Legacy/simple-GA parameters (fallback) |
| `models/ml_model.json` | Trained linear ML coefficients |
| `models/model_metadata.json` | Version history and metadata |
| `models/best_fitness_tracker.json` | All-time best fitness tracker |
| `models/archive/` | Archived model versions (up to 20) |
| `production_signals.csv` | Every production-loop decision with reasons |
| `learning_log.csv`, `performance_log.csv` | Online-learning trade history and stats |
| `trading_bot_mt5.log`, `ga_framework.log`, `production_loop.log`, `learning.log`, `versioning.log` | Runtime logs (git-ignored) |

---

## 🧭 Roadmap Ideas

- Live regime detection feed for the production loop
- LLM-assisted trade journaling via the local Ollama (`qwen-3b`) endpoint
- Multi-symbol support (ETHUSD, XAUUSD)
- Web dashboard for remote monitoring
- Dockerized dry-run CI backtest on every push

---

## 📜 License

Provided **as-is, without warranty of any kind**. This project is intended for educational purposes. You are fully responsible for complying with your local laws and your broker's terms of service when using automated trading software.
