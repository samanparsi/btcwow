# Hybrid RL-Expert Trading System (BTC-USD)

[![License: Restricted](https://img.shields.io/badge/License-Restricted-red.svg)](6ab3cbb4)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An advanced algorithmic trading framework that combines Deep Learning, Ensemble Methods, and Reinforcement Learning to trade Bitcoin volatility.

## 🚀 Project Overview
This system implements a decoupled architecture where feature extraction is handled by **Supervised Experts** and capital allocation is managed by a **PPO Reinforcement Learning** agent. This approach reduces the 'noise' usually associated with pure RL trading models.

## 🛠 Architecture

### 1. The Expert Layer
- **CNN-LSTM Network**: Processes OHLCV data to extract spatial patterns and temporal dependencies.
- **Random Forest Regressor**: Provides a non-linear statistical baseline for price action residuals.
- **Regime Classifier**: Automatically segments market conditions into *High Volatility* vs. *Stable* states.

### 2. The Decision Layer
- **PPO Agent**: A Stable Baselines3 implementation that maps expert signals into actions.
- **Bi-Directional Trading**: Supports Long (+2x, +1x), Neutral (0), and Short (-1x, -2x) positions.

### 3. Safety & Risk
- **ATR-Dynamic Stops**: Adaptive Stop-Loss and Take-Profit based on current market volatility.
- **Realistic Backtesting**: Built-in 0.1% transaction fee modeling and slippage simulation.

## 📊 Performance
| Metric | Value |
| :--- | :--- |
| **Asset** | BTC-USD |
| **Timeframe** | Daily (1D) |
| **Final ROI** | ~440% (Based on recent backtest) |
| **Validation** | 20% Out-of-Sample Walk-Forward |

## 💻 Quick Start
```bash
pip install stable-baselines3 gymnasium yfinance torch scikit-learn
```
*Refer to the implementation cell below for the full source code.*
