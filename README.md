# BTC WOW – Bitcoin 4H Trading Strategy Backtester

A disciplined, systematic approach to predict and trade Bitcoin price movements on the 4-hour timeframe.

## Strategy Overview

**Core Concept:** Multi-timeframe bias (daily trend) + 4H entry signals + strict risk management.

### Key Components

1. **Multi-Timeframe Bias**
   - Daily trend detection: price vs. 200 EMA
   - Only take long trades if daily bias is bullish

2. **4H Entry Signals**
   - Trend-following: 50 EMA crossover, MACD bullish
   - Volume confirmation
   - Price action at support/resistance

3. **Risk Management**
   - Fixed risk per trade: 0.5% of account equity
   - Stop loss: below recent swing low or 1.5× ATR
   - Target: 2× risk or trailing stop

4. **Execution**
   - Realistic fees, slippage, and bid-ask spread included
   - Walk-forward backtesting to avoid overfitting

## Project Structure

```
btcwow/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── config.py              # Configuration constants
├── data_fetcher.py        # Fetch 4H BTC OHLCV data
├── indicators.py          # Technical indicators (EMA, RSI, MACD, ATR, etc.)
├── strategy.py            # Trading strategy logic
├── backtest.py            # Backtesting engine
├── main.py                # Entry point
└── data/                  # Local cache of downloaded data
    └── btc_4h.csv         # Historical 4H OHLCV data
```

## Getting Started

### Installation

```bash
# Clone the repo
git clone https://github.com/samanparsi/btcwow.git
cd btcwow

# Install dependencies
pip install -r requirements.txt
```

### Run a Backtest

```bash
# Download data and run backtest
python main.py
```

This will:
- Fetch 4H BTC/USDT data (default: last 2 years)
- Run the strategy using walk-forward validation
- Print performance metrics (return, Sharpe, max drawdown, etc.)
- Save results to `results.json`

## Configuration

Edit `config.py` to customize:
- Date ranges
- Starting capital
- Risk per trade
- Fee/slippage assumptions
- Indicator parameters

## Expected Metrics

A robust 4H strategy on BTC typically shows:
- **Win Rate:** 40–60%
- **Profit Factor:** > 1.5 (avg win / avg loss)
- **Sharpe Ratio:** > 1.0
- **Max Drawdown:** 10–30%
- **Expectancy:** Positive (avg win − avg loss > 0)

## Next Steps

1. Backtest current strategy
2. Optimize parameters (walk-forward)
3. Add forward-testing with live alerts
4. Integrate with exchange API for live trading (with proper risk controls)

## Disclaimer

This is an educational tool. Cryptocurrency trading carries risk. Always:
- Backtest extensively
- Start with small size
- Use proper risk management
- Never risk more than you can afford to lose
