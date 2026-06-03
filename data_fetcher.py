"""
Data Fetcher Module
Downloads and manages 4H BTC/USDT data from Binance or local cache.
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Optional
import requests


class DataFetcher:
    """Fetch and manage Bitcoin OHLCV data."""
    
    BINANCE_API = "https://api.binance.com/api/v3"
    
    def __init__(self, symbol: str = "BTCUSDT", timeframe: str = "4h", cache_dir: str = "data"):
        """
        Initialize DataFetcher.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            timeframe: Timeframe (e.g., 4h, 1h, 1d)
            cache_dir: Directory to cache downloaded data
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f"{symbol.lower()}_{timeframe}.csv")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def fetch_binance_klines(
        self,
        start_date: str,
        end_date: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical klines from Binance API.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Records per request (max 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"[DataFetcher] Fetching {self.symbol} {self.timeframe} from Binance...")
        
        # Convert dates to milliseconds
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_klines = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                params = {
                    'symbol': self.symbol,
                    'interval': self.timeframe,
                    'startTime': current_ts,
                    'limit': limit
                }
                
                response = requests.get(
                    f"{self.BINANCE_API}/klines",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                
                klines = response.json()
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Move to next batch
                current_ts = klines[-1][0] + 1
                print(f"  Downloaded {len(klines)} candles, ending at {datetime.fromtimestamp(klines[-1][0]/1000)}")
                
            except Exception as e:
                print(f"[ERROR] Failed to fetch data: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_klines,
            columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                     'close_time', 'quote_volume', 'trades', 'buy_quote_volume', 'ignore']
        )
        
        # Clean and convert types
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.astype({
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': float
        })
        
        df.set_index('datetime', inplace=True)
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        print(f"[DataFetcher] Downloaded {len(df)} candles ({df.index[0]} to {df.index[-1]})")
        
        return df
    
    def load_from_cache(self) -> Optional[pd.DataFrame]:
        """Load data from local cache if it exists."""
        if os.path.exists(self.cache_file):
            try:
                df = pd.read_csv(self.cache_file, index_col='datetime', parse_dates=True)
                print(f"[DataFetcher] Loaded {len(df)} candles from cache: {self.cache_file}")
                return df
            except Exception as e:
                print(f"[WARNING] Failed to load cache: {e}")
        return None
    
    def save_to_cache(self, df: pd.DataFrame) -> None:
        """Save data to local cache."""
        try:
            df.to_csv(self.cache_file)
            print(f"[DataFetcher] Saved {len(df)} candles to cache: {self.cache_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save cache: {e}")
    
    def get_data(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Get OHLCV data, using cache if available.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Use cached data if available
            force_download: Force download even if cache exists
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try to load from cache
        if use_cache and not force_download:
            cached_df = self.load_from_cache()
            if cached_df is not None:
                # Filter to requested date range
                mask = (cached_df.index >= start_date) & (cached_df.index <= end_date)
                filtered_df = cached_df[mask]
                
                if len(filtered_df) > 0:
                    return filtered_df
        
        # Download from Binance
        df = self.fetch_binance_klines(start_date, end_date)
        
        # Save to cache
        if use_cache:
            self.save_to_cache(df)
        
        return df
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> bool:
        """
        Validate OHLCV dataframe structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        has_cols = required_cols.issubset(df.columns)
        
        if not has_cols:
            print(f"[ERROR] Missing columns. Required: {required_cols}, Got: {set(df.columns)}")
            return False
        
        # Check for NaN values
        if df.isnull().any().any():
            print(f"[WARNING] DataFrame contains NaN values")
            return False
        
        # Check OHLC ordering
        invalid_ohlc = (df['high'] < df['low']).any() or \
                       (df['high'] < df['open']).any() or \
                       (df['high'] < df['close']).any()
        
        if invalid_ohlc:
            print(f"[ERROR] Invalid OHLC values detected")
            return False
        
        return True
