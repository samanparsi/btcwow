"""
Strategy Module
Defines the 4H BTC trading strategy logic (entry, exit, position management).
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class TradeSignal(Enum):
    """Trade signal types."""
    NO_SIGNAL = 0
    LONG_ENTRY = 1
    LONG_EXIT = 2
    SHORT_ENTRY = 3
    SHORT_EXIT = 4


@dataclass
class Trade:
    """Represents a single trade."""
    entry_idx: int
    entry_price: float
    entry_time: pd.Timestamp
    
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None
    
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    position_size: float = 0.0  # In BTC
    pnl: float = 0.0  # Realized P&L in USDT
    pnl_pct: float = 0.0  # P&L %
    
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_idx is None
    
    def close(self, exit_idx: int, exit_price: float, exit_time: pd.Timestamp, reason: str):
        """Close the trade."""
        self.exit_idx = exit_idx
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        
        # Calculate P&L
        self.pnl = (self.exit_price - self.entry_price) * self.position_size
        self.pnl_pct = ((self.exit_price - self.entry_price) / self.entry_price) * 100


class Strategy:
    """4H BTC Trading Strategy."""
    
    def __init__(self, config: dict):
        """
        Initialize strategy with config.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.trades = []
        self.current_trade: Optional[Trade] = None
    
    def check_bias(self, df_daily: pd.DataFrame, idx: int) -> str:
        """
        Check daily bias (trend direction).
        
        Args:
            df_daily: Daily timeframe dataframe
            idx: Index in daily dataframe
            
        Returns:
            'LONG', 'SHORT', or 'NEUTRAL'
        """
        if idx < 1:
            return 'NEUTRAL'
        
        if pd.isna(df_daily.iloc[idx]['ema_slow']):
            return 'NEUTRAL'
        
        close = df_daily.iloc[idx]['close']
        ema_slow = df_daily.iloc[idx]['ema_slow']
        
        if close > ema_slow:
            return 'LONG'
        elif close < ema_slow:
            return 'SHORT'
        else:
            return 'NEUTRAL'
    
    def generate_entry_signal(self, df: pd.DataFrame, idx: int, bias: str) -> TradeSignal:
        """
        Generate entry signal based on 4H indicators.
        
        Args:
            df: 4H dataframe with indicators
            idx: Current index
            bias: Daily bias ('LONG', 'SHORT', 'NEUTRAL')
            
        Returns:
            TradeSignal
        """
        if idx < 2 or pd.isna(df.iloc[idx]['close']):
            return TradeSignal.NO_SIGNAL
        
        if self.current_trade is not None:
            return TradeSignal.NO_SIGNAL  # Already in trade
        
        current_row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]
        
        ema_fast = current_row['ema_fast']
        ema_slow = current_row['ema_slow']
        macd = current_row['macd']
        macd_signal = current_row['macd_signal']
        macd_prev = df.iloc[idx - 1]['macd']
        macd_signal_prev = df.iloc[idx - 1]['macd_signal']
        rsi = current_row['rsi']
        volume = current_row['volume']
        volume_ma = current_row['volume_ma']
        close = current_row['close']
        
        # Skip if indicators not ready
        if pd.isna([ema_fast, ema_slow, macd, rsi, volume_ma]).any():
            return TradeSignal.NO_SIGNAL
        
        # ===== LONG ENTRY CONDITIONS =====
        if bias == 'LONG':
            # Condition 1: 50 EMA > 200 EMA (trend alignment)
            ema_bullish = ema_fast > ema_slow
            
            # Condition 2: MACD bullish crossover
            macd_crossover = (macd_prev <= macd_signal_prev) and (macd > macd_signal)
            
            # Condition 3: RSI in favorable zone
            rsi_valid = self.config.get('ENTRY_RSI_MIN', 30) < rsi < self.config.get('ENTRY_RSI_MAX', 55)
            
            # Condition 4: Volume confirmation
            volume_valid = volume >= volume_ma * self.config.get('VOLUME_MIN_RATIO', 1.0)
            
            # Condition 5: Price pullback to 50 EMA (optional)
            price_at_ema = abs(close - ema_fast) < abs(ema_fast - ema_slow) * 0.02
            
            if ema_bullish and macd_crossover and rsi_valid and volume_valid:
                return TradeSignal.LONG_ENTRY
        
        # ===== SHORT ENTRY CONDITIONS =====
        elif bias == 'SHORT':
            # Mirror conditions for short
            ema_bearish = ema_fast < ema_slow
            macd_crossover = (macd_prev >= macd_signal_prev) and (macd < macd_signal)
            rsi_valid = (100 - self.config.get('ENTRY_RSI_MAX', 55)) < rsi < (100 - self.config.get('ENTRY_RSI_MIN', 30))
            volume_valid = volume >= volume_ma * self.config.get('VOLUME_MIN_RATIO', 1.0)
            
            if ema_bearish and macd_crossover and rsi_valid and volume_valid:
                return TradeSignal.SHORT_ENTRY
        
        return TradeSignal.NO_SIGNAL
    
    def generate_exit_signal(self, df: pd.DataFrame, idx: int) -> TradeSignal:
        """
        Generate exit signal for open trade.
        
        Args:
            df: 4H dataframe
            idx: Current index
            
        Returns:
            TradeSignal (LONG_EXIT or SHORT_EXIT) or NO_SIGNAL
        """
        if self.current_trade is None:
            return TradeSignal.NO_SIGNAL
        
        if idx < 1 or pd.isna(df.iloc[idx]['close']):
            return TradeSignal.NO_SIGNAL
        
        current_price = df.iloc[idx]['close']
        
        # Check stop loss
        if current_price <= self.current_trade.stop_loss:
            return TradeSignal.LONG_EXIT if self.current_trade.position_size > 0 else TradeSignal.SHORT_EXIT
        
        # Check take profit
        if current_price >= self.current_trade.take_profit:
            return TradeSignal.LONG_EXIT if self.current_trade.position_size > 0 else TradeSignal.SHORT_EXIT
        
        # Check trailing stop
        if self.config.get('USE_TRAILING_STOP', True):
            atr = df.iloc[idx]['atr']
            if pd.notna(atr):
                trailing_stop = current_price - (atr * self.config.get('ATR_TRAIL_MULTIPLE', 1.0))
                
                if self.current_trade.position_size > 0:  # Long
                    if current_price < trailing_stop:
                        return TradeSignal.LONG_EXIT
                else:  # Short
                    trailing_stop = current_price + (atr * self.config.get('ATR_TRAIL_MULTIPLE', 1.0))
                    if current_price > trailing_stop:
                        return TradeSignal.SHORT_EXIT
        
        return TradeSignal.NO_SIGNAL
    
    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss: float
    ) -> float:
        """
        Calculate position size based on risk management.
        
        Args:
            account_equity: Current account equity
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Position size in BTC
        """
        risk_amount = account_equity * self.config.get('RISK_PER_TRADE', 0.005)
        
        distance_to_stop = abs(entry_price - stop_loss)
        if distance_to_stop == 0:
            return 0
        
        position_size = risk_amount / distance_to_stop
        
        # Hard cap on position size
        max_position = account_equity * self.config.get('MAX_POSITION_SIZE', 0.10) / entry_price
        position_size = min(position_size, max_position)
        
        return position_size
    
    def execute_long_entry(
        self,
        df: pd.DataFrame,
        idx: int,
        account_equity: float,
        fees: float = 0.001
    ) -> Optional[Trade]:
        """Execute a long entry trade."""
        current_row = df.iloc[idx]
        entry_price = current_row['close']
        atr = current_row['atr']
        
        if pd.isna(atr):
            return None
        
        # Calculate stop loss
        stop_loss = entry_price - (atr * self.config.get('ATR_STOP_MULTIPLE', 1.5))
        
        # Calculate position size
        position_size = self.calculate_position_size(account_equity, entry_price, stop_loss)
        
        if position_size <= 0:
            return None
        
        # Calculate take profit
        reward_distance = (entry_price - stop_loss) * self.config.get('REWARD_RISK_RATIO', 2.0)
        take_profit = entry_price + reward_distance
        
        # Create trade
        trade = Trade(
            entry_idx=idx,
            entry_price=entry_price,
            entry_time=df.index[idx],
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size
        )
        
        self.current_trade = trade
        return trade
    
    def execute_exit(
        self,
        df: pd.DataFrame,
        idx: int,
        reason: str
    ) -> Optional[Trade]:
        """Execute trade exit."""
        if self.current_trade is None:
            return None
        
        exit_price = df.iloc[idx]['close']
        closed_trade = self.current_trade
        closed_trade.close(idx, exit_price, df.index[idx], reason)
        
        self.trades.append(closed_trade)
        self.current_trade = None
        
        return closed_trade
