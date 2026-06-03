"""
Configuration constants for BTC 4H trading strategy.
Adjust these to customize your backtest parameters.
"""

import os
from datetime import datetime, timedelta

# ============================================================================
# DATA & TIME SETTINGS
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
BTCUSDT_CSV = os.path.join(DATA_DIR, 'btc_4h.csv')

# Date range for backtesting
START_DATE = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')  # 2 years back
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Data source (Binance API, Yahoo Finance, or local CSV)
DATA_SOURCE = 'binance'  # 'binance', 'yahoo', 'local'
SYMBOL = 'BTCUSDT'
TIMEFRAME = '4h'

# ============================================================================
# ACCOUNT & RISK MANAGEMENT
# ============================================================================
INITIAL_CAPITAL = 10000  # Starting balance in USDT
RISK_PER_TRADE = 0.005  # 0.5% of account equity per trade
MAX_POSITION_SIZE = 0.10  # Max 10% of capital per trade (optional hard cap)

# ============================================================================
# FEES & SLIPPAGE (realistic assumptions)
# ============================================================================
TAKER_FEE = 0.001  # 0.1% taker fee (Binance default)
MAKER_FEE = 0.001  # 0.1% maker fee
SLIPPAGE_BPS = 2   # 2 basis points (0.02%) for entry/exit slippage
SPREAD_BPS = 1     # 1 basis point bid-ask spread on entry

# ============================================================================
# INDICATOR PARAMETERS
# ============================================================================
# EMA (Exponential Moving Average)
EMA_FAST = 50
EMA_SLOW = 200

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# RSI (Relative Strength Index)
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# ATR (Average True Range) for volatility-based stops
ATR_PERIOD = 14
ATR_STOP_MULTIPLE = 1.5  # Stop loss = entry - (ATR * 1.5)
ATR_TRAIL_MULTIPLE = 1.0  # Trailing stop = close - (ATR * 1.0)

# Bollinger Bands (optional for mean reversion)
BB_PERIOD = 20
BB_STD_DEV = 2

# Volume
VOLUME_MA_PERIOD = 20  # 20-period volume moving average

# ============================================================================
# STRATEGY PARAMETERS
# ============================================================================
# Trend bias (must be above this EMA on daily for long trades)
BIAS_EMA_DAILY = 200

# 4H Entry conditions
ENTRY_RSI_MIN = 30  # Don't enter if RSI too low
ENTRY_RSI_MAX = 55  # Don't enter if RSI too high

# Volume confirmation
VOLUME_MIN_RATIO = 1.0  # Current volume >= avg volume * 1.0

# ============================================================================
# POSITION MANAGEMENT
# ============================================================================
REWARD_RISK_RATIO = 2.0  # Target = entry + (stop_loss_distance * 2.0)
USE_TRAILING_STOP = True  # Use ATR-based trailing stop instead of fixed target

# Max drawdown circuit breaker (pause trading if exceeded)
MAX_DAILY_DRAWDOWN_PCT = 5.0  # Stop trading if down 5% in a day

# ============================================================================
# BACKTEST ENGINE SETTINGS
# ============================================================================
# Walk-forward testing
WALKFORWARD_WINDOW_DAYS = 90   # Train on 90 days
WALKFORWARD_STEP_DAYS = 30     # Step forward 30 days (30% overlap for stability)
WALKFORWARD_TEST_DAYS = 30     # Test on 30 days out-of-sample

# Allow overlap for continuous testing
ALLOW_CONCURRENT_TRADES = False  # One trade at a time
MAX_TRADES_PER_DAY = 3           # Max entries per day to avoid overtrading

# ============================================================================
# OUTPUT & REPORTING
# ============================================================================
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'results.json')
VERBOSE = True  # Print detailed logs during backtest
PLOT_RESULTS = False  # Generate matplotlib charts (requires matplotlib)

# ============================================================================
# MACHINE LEARNING (optional, future expansion)
# ============================================================================
ML_ENABLED = False
ML_MODEL = 'xgboost'  # 'xgboost', 'lightgbm', 'neural_net'
ML_FEATURE_ENGINEERING = False
