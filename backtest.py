"""
Backtesting Engine
Simulates strategy on historical data with realistic execution assumptions.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import sys

from strategy import Strategy, TradeSignal
from indicators import add_all_indicators


class BacktestEngine:
    """Backtesting engine with walk-forward validation."""
    
    def __init__(self, config: dict):
        """Initialize backtester."""
        self.config = config
        self.results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'expectancy': 0.0,
            'sharpe_ratio': 0.0,
            'final_balance': config.get('INITIAL_CAPITAL', 10000),
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'trades': []
        }
    
    def simulate(
        self,
        df_4h: pd.DataFrame,
        df_daily: pd.DataFrame,
        initial_balance: float,
        taker_fee: float = 0.001,
        slippage_bps: float = 2,
        verbose: bool = True
    ) -> Dict:
        """
        Simulate strategy on historical data.
        
        Args:
            df_4h: 4H OHLCV dataframe with indicators
            df_daily: Daily OHLCV dataframe with indicators
            initial_balance: Starting balance in USDT
            taker_fee: Taker fee rate
            slippage_bps: Slippage in basis points
            verbose: Print detailed logs
            
        Returns:
            Results dictionary with performance metrics
        """
        if verbose:
            print(f"\n[Backtest] Starting simulation...")
            print(f"  Data range: {df_4h.index[0]} to {df_4h.index[-1]}")
            print(f"  Initial balance: ${initial_balance:,.2f}")
            print(f"  Taker fee: {taker_fee*100:.2f}%, Slippage: {slippage_bps} bps")
        
        strategy = Strategy(self.config)
        
        balance = initial_balance
        balance_history = [balance]
        equity_history = [balance]
        
        trades_closed = 0
        
        # Find daily index mapping
        daily_dates = df_daily.index
        
        # Main trading loop
        for idx in range(len(df_4h)):
            current_time = df_4h.index[idx]
            
            # Find corresponding daily candle
            daily_idx = daily_dates.searchsorted(current_time)
            if daily_idx >= len(daily_dates):
                daily_idx = len(daily_dates) - 1
            
            # Get daily bias
            bias = strategy.check_bias(df_daily, daily_idx)
            
            # Generate entry signal
            if strategy.current_trade is None:
                entry_signal = strategy.generate_entry_signal(df_4h, idx, bias)
                
                if entry_signal == TradeSignal.LONG_ENTRY:
                    trade = strategy.execute_long_entry(df_4h, idx, balance, taker_fee)
                    
                    if trade is not None:
                        # Calculate entry cost with fees
                        entry_cost = trade.entry_price * trade.position_size * (1 + taker_fee)
                        entry_cost += entry_cost * (slippage_bps / 10000)
                        
                        if verbose and trades_closed < 10:  # Print first 10 trades
                            print(f"\n[Trade {trades_closed + 1}] LONG ENTRY at {current_time}")
                            print(f"  Price: ${trade.entry_price:,.2f}, Size: {trade.position_size:.4f} BTC")
                            print(f"  Stop: ${trade.stop_loss:,.2f}, Target: ${trade.take_profit:,.2f}")
                            print(f"  Bias: {bias}")
            
            # Generate exit signal
            else:
                exit_signal = strategy.generate_exit_signal(df_4h, idx)
                
                if exit_signal == TradeSignal.LONG_EXIT:
                    trade = strategy.execute_exit(df_4h, idx, exit_signal.name)
                    
                    if trade is not None:
                        # Calculate exit proceeds with fees
                        exit_proceeds = trade.exit_price * trade.position_size * (1 - taker_fee)
                        exit_proceeds -= exit_proceeds * (slippage_bps / 10000)
                        
                        # Update balance
                        entry_cost = trade.entry_price * trade.position_size * (1 + taker_fee)
                        entry_cost += entry_cost * (slippage_bps / 10000)
                        
                        pnl = exit_proceeds - entry_cost
                        balance += pnl
                        
                        trades_closed += 1
                        
                        if verbose and trades_closed <= 10:
                            print(f"  EXIT at {trade.exit_time}: ${trade.exit_price:,.2f}")
                            print(f"  P&L: ${pnl:,.2f} ({(pnl/entry_cost)*100:.2f}%)")
                            print(f"  Balance: ${balance:,.2f}")
            
            # Update equity (mark-to-market open trades)
            current_equity = balance
            if strategy.current_trade is not None:
                current_price = df_4h.iloc[idx]['close']
                mtm = (current_price - strategy.current_trade.entry_price) * strategy.current_trade.position_size
                current_equity = balance + mtm
            
            balance_history.append(balance)
            equity_history.append(current_equity)
        
        # Close any open trades at the end
        if strategy.current_trade is not None:
            trade = strategy.execute_exit(df_4h, len(df_4h) - 1, "END_OF_DATA")
            if trade is not None:
                exit_proceeds = trade.exit_price * trade.position_size * (1 - taker_fee)
                entry_cost = trade.entry_price * trade.position_size * (1 + taker_fee)
                pnl = exit_proceeds - entry_cost
                balance += pnl
        
        # Calculate metrics
        self._calculate_metrics(strategy.trades, balance, initial_balance, equity_history, verbose)
        
        return self.results
    
    def _calculate_metrics(self, trades: List, final_balance: float, initial_balance: float, equity_history: List, verbose: bool = True):
        """Calculate performance metrics."""
        
        self.results['total_trades'] = len(trades)
        self.results['final_balance'] = final_balance
        self.results['total_return'] = final_balance - initial_balance
        self.results['total_return_pct'] = (self.results['total_return'] / initial_balance) * 100
        
        if len(trades) == 0:
            if verbose:
                print("\n[Results] No trades executed")
            return
        
        # Separate winning and losing trades
        pnls = [trade.pnl for trade in trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        self.results['winning_trades'] = len(winning_trades)
        self.results['losing_trades'] = len(losing_trades)
        self.results['win_rate'] = (len(winning_trades) / len(trades)) * 100 if trades else 0
        
        self.results['total_profit'] = sum(winning_trades) if winning_trades else 0
        self.results['total_loss'] = sum(losing_trades) if losing_trades else 0
        
        avg_win = self.results['total_profit'] / len(winning_trades) if winning_trades else 0
        avg_loss = abs(self.results['total_loss']) / len(losing_trades) if losing_trades else 0
        
        self.results['avg_profit'] = avg_win
        self.results['avg_loss'] = avg_loss
        self.results['profit_factor'] = avg_win / avg_loss if avg_loss > 0 else 0
        self.results['expectancy'] = ((self.results['win_rate'] / 100) * avg_win) - ((1 - self.results['win_rate'] / 100) * avg_loss)
        
        # Drawdown
        equity_array = np.array(equity_history)
        peak = np.maximum.accumulate(equity_array)
        drawdown = peak - equity_array
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        max_drawdown_pct = (max_drawdown / initial_balance) * 100 if initial_balance > 0 else 0
        
        self.results['max_drawdown'] = max_drawdown
        self.results['max_drawdown_pct'] = max_drawdown_pct
        
        # Sharpe ratio (simplified, assuming daily returns)
        returns = np.diff(equity_array) / equity_array[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) > 0:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)  # Annualized
            self.results['sharpe_ratio'] = sharpe
        
        # Store trade details
        self.results['trades'] = [
            {
                'entry_time': str(trade.entry_time),
                'entry_price': round(trade.entry_price, 2),
                'exit_time': str(trade.exit_time),
                'exit_price': round(trade.exit_price, 2),
                'pnl': round(trade.pnl, 2),
                'pnl_pct': round(trade.pnl_pct, 2)
            }
            for trade in trades[:100]  # Store first 100 trades
        ]
        
        if verbose:
            print("\n" + "="*60)
            print("BACKTEST RESULTS")
            print("="*60)
            print(f"Total Trades:       {self.results['total_trades']}")
            print(f"Winning Trades:     {self.results['winning_trades']} ({self.results['win_rate']:.1f}%)")
            print(f"Losing Trades:      {self.results['losing_trades']}")
            print(f"Total Profit:       ${self.results['total_profit']:,.2f}")
            print(f"Total Loss:         ${self.results['total_loss']:,.2f}")
            print(f"Profit Factor:      {self.results['profit_factor']:.2f}")
            print(f"Avg Win:            ${self.results['avg_profit']:,.2f}")
            print(f"Avg Loss:           ${self.results['avg_loss']:,.2f}")
            print(f"Expectancy:         ${self.results['expectancy']:,.2f}")
            print(f"\nFinal Balance:      ${self.results['final_balance']:,.2f}")
            print(f"Total Return:       ${self.results['total_return']:,.2f} ({self.results['total_return_pct']:.2f}%)")
            print(f"Max Drawdown:       ${self.results['max_drawdown']:,.2f} ({self.results['max_drawdown_pct']:.2f}%)")
            print(f"Sharpe Ratio:       {self.results['sharpe_ratio']:.2f}")
            print("="*60)
