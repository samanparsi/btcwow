"""
Main Entry Point
Orchestrates data fetching, indicator calculation, and backtesting.
"""

import json
import sys
import pandas as pd
import config
from data_fetcher import DataFetcher
from indicators import add_all_indicators
from backtest import BacktestEngine


def resample_to_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 4H data to daily OHLCV.
    
    Args:
        df_4h: 4H OHLCV dataframe
        
    Returns:
        Daily OHLCV dataframe
    """
    daily = pd.DataFrame()
    daily['open'] = df_4h['open'].resample('D').first()
    daily['high'] = df_4h['high'].resample('D').max()
    daily['low'] = df_4h['low'].resample('D').min()
    daily['close'] = df_4h['close'].resample('D').last()
    daily['volume'] = df_4h['volume'].resample('D').sum()
    
    # Remove NaN rows
    daily = daily.dropna()
    
    return daily


def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("BTC WOW - 4H Bitcoin Trading Strategy Backtester")
    print("="*70)
    
    # ========== STEP 1: FETCH DATA ==========
    print("\n[STEP 1] Fetching market data...")
    
    fetcher = DataFetcher(
        symbol=config.SYMBOL,
        timeframe=config.TIMEFRAME,
        cache_dir=config.DATA_DIR
    )
    
    df_4h = fetcher.get_data(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        use_cache=True,
        force_download=False
    )
    
    if not DataFetcher.validate_ohlcv(df_4h):
        print("[ERROR] Invalid data. Exiting.")
        sys.exit(1)
    
    print(f"✓ Loaded {len(df_4h)} 4H candles")
    
    # ========== STEP 2: CALCULATE INDICATORS ==========
    print("\n[STEP 2] Calculating technical indicators...")
    
    # Convert config to dict for indicators
    config_dict = {
        'EMA_FAST': config.EMA_FAST,
        'EMA_SLOW': config.EMA_SLOW,
        'MACD_FAST': config.MACD_FAST,
        'MACD_SLOW': config.MACD_SLOW,
        'MACD_SIGNAL': config.MACD_SIGNAL,
        'RSI_PERIOD': config.RSI_PERIOD,
        'ATR_PERIOD': config.ATR_PERIOD,
        'BB_PERIOD': config.BB_PERIOD,
        'BB_STD_DEV': config.BB_STD_DEV,
        'VOLUME_MA_PERIOD': config.VOLUME_MA_PERIOD,
    }
    
    df_4h = add_all_indicators(df_4h, config_dict)
    print(f"✓ Added {len([c for c in df_4h.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} indicators")
    
    # Resample to daily for bias calculation
    print("  Resampling to daily data for trend bias...")
    df_daily = resample_to_daily(df_4h)
    df_daily = add_all_indicators(df_daily, config_dict)
    print(f"✓ Created {len(df_daily)} daily candles")
    
    # ========== STEP 3: BACKTEST ==========
    print("\n[STEP 3] Running backtest simulation...")
    
    backtest_engine = BacktestEngine(config_dict)
    
    results = backtest_engine.simulate(
        df_4h=df_4h,
        df_daily=df_daily,
        initial_balance=config.INITIAL_CAPITAL,
        taker_fee=config.TAKER_FEE,
        slippage_bps=config.SLIPPAGE_BPS,
        verbose=config.VERBOSE
    )
    
    # ========== STEP 4: SAVE RESULTS ==========
    print("\n[STEP 4] Saving results...")
    
    try:
        with open(config.RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Results saved to: {config.RESULTS_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)
    print(f"Initial Balance:    ${config.INITIAL_CAPITAL:,.2f}")
    print(f"Final Balance:      ${results['final_balance']:,.2f}")
    print(f"Total Return:       ${results['total_return']:,.2f} ({results['total_return_pct']:.2f}%)")
    print(f"\nTotal Trades:       {results['total_trades']}")
    print(f"Win Rate:           {results['win_rate']:.1f}%")
    print(f"Profit Factor:      {results['profit_factor']:.2f}")
    print(f"Expectancy:         ${results['expectancy']:,.2f}/trade")
    print(f"Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:       {results['max_drawdown_pct']:.2f}%")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()
