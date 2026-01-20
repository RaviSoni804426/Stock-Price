"""
Stock data fetcher module.
Fetches historical OHLCV data from Yahoo Finance for NSE stocks.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging

import sys
sys.path.insert(0, str(__file__).rsplit("\\", 2)[0])
from config import MIN_HISTORY_YEARS, CACHE_DIR

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Fetches historical stock data from Yahoo Finance.
    Handles NSE stock symbol formatting and data retrieval.
    """
    
    # Common NSE stock symbols mapping (without .NS suffix)
    NSE_SYMBOLS = {
        "INFOSYS": "INFY.NS",
        "INFY": "INFY.NS",
        "TCS": "TCS.NS",
        "RELIANCE": "RELIANCE.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "HDFC": "HDFCBANK.NS",
        "ICICIBANK": "ICICIBANK.NS",
        "SBIN": "SBIN.NS",
        "BHARTIARTL": "BHARTIARTL.NS",
        "ITC": "ITC.NS",
        "KOTAKBANK": "KOTAKBANK.NS",
        "LT": "LT.NS",
        "HINDUNILVR": "HINDUNILVR.NS",
        "AXISBANK": "AXISBANK.NS",
        "BAJFINANCE": "BAJFINANCE.NS",
        "MARUTI": "MARUTI.NS",
        "WIPRO": "WIPRO.NS",
        "HCLTECH": "HCLTECH.NS",
        "ASIANPAINT": "ASIANPAINT.NS",
        "TITAN": "TITAN.NS",
        "SUNPHARMA": "SUNPHARMA.NS",
        "ULTRACEMCO": "ULTRACEMCO.NS",
        "NESTLEIND": "NESTLEIND.NS",
        "TATAMOTORS": "TATAMOTORS.NS",
        "TATASTEEL": "TATASTEEL.NS",
        "POWERGRID": "POWERGRID.NS",
        "NTPC": "NTPC.NS",
        "ONGC": "ONGC.NS",
        "JSWSTEEL": "JSWSTEEL.NS",
        "M&M": "M&M.NS",
        "ADANIENT": "ADANIENT.NS",
        "ADANIPORTS": "ADANIPORTS.NS",
        "COALINDIA": "COALINDIA.NS",
        "DRREDDY": "DRREDDY.NS",
        "CIPLA": "CIPLA.NS",
        "DIVISLAB": "DIVISLAB.NS",
        "EICHERMOT": "EICHERMOT.NS",
        "GRASIM": "GRASIM.NS",
        "HEROMOTOCO": "HEROMOTOCO.NS",
        "HINDALCO": "HINDALCO.NS",
        "INDUSINDBK": "INDUSINDBK.NS",
        "TECHM": "TECHM.NS",
        "BRITANNIA": "BRITANNIA.NS",
        "BAJAJFINSV": "BAJAJFINSV.NS",
        "APOLLOHOSP": "APOLLOHOSP.NS",
        "BPCL": "BPCL.NS",
        "SBILIFE": "SBILIFE.NS",
        "HDFCLIFE": "HDFCLIFE.NS",
        "NIFTY": "^NSEI",  # Nifty 50 index
        "BANKNIFTY": "^NSEBANK",  # Bank Nifty index
    }
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the data fetcher.
        
        Args:
            use_cache: Whether to cache downloaded data locally
        """
        self.use_cache = use_cache
        self.cache_dir = CACHE_DIR
        
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Convert user-friendly symbol to Yahoo Finance format.
        
        Args:
            symbol: Stock symbol (e.g., "INFOSYS", "TCS", "RELIANCE")
            
        Returns:
            Yahoo Finance compatible symbol (e.g., "INFY.NS")
        """
        symbol = symbol.upper().strip()
        
        # Check if it's in our mapping
        if symbol in self.NSE_SYMBOLS:
            return self.NSE_SYMBOLS[symbol]
        
        # If already has .NS suffix, return as-is
        if symbol.endswith(".NS"):
            return symbol
            
        # Otherwise, append .NS for NSE stocks
        return f"{symbol}.NS"
    
    def _get_cache_path(self, symbol: str) -> str:
        """Get the cache file path for a symbol."""
        safe_symbol = symbol.replace(".", "_").replace("^", "_")
        return self.cache_dir / f"{safe_symbol}_daily.parquet"
    
    def fetch(
        self, 
        symbol: str, 
        years: int = MIN_HISTORY_YEARS,
        end_date: Optional[datetime] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch historical daily OHLCV data for a stock.
        
        Args:
            symbol: Stock symbol (e.g., "INFOSYS", "TCS")
            years: Number of years of history to fetch
            end_date: End date for data (defaults to today)
            
        Returns:
            Tuple of (DataFrame with OHLCV data, error message if any)
            DataFrame columns: Open, High, Low, Close, Volume, Adj Close
        """
        yahoo_symbol = self._normalize_symbol(symbol)
        
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=years * 365)
        
        logger.info(f"Fetching data for {yahoo_symbol} from {start_date.date()} to {end_date.date()}")
        
        try:
            # Try to load from cache first
            cache_path = self._get_cache_path(yahoo_symbol)
            
            if self.use_cache and cache_path.exists():
                cached_data = pd.read_parquet(cache_path)
                cache_end = cached_data.index.max()
                
                # If cache is recent enough (within 1 day), use it
                if (end_date.date() - cache_end.date()).days <= 1:
                    logger.info(f"Using cached data for {yahoo_symbol}")
                    return self._filter_date_range(cached_data, start_date, end_date), None
            
            # Fetch from Yahoo Finance
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                return None, f"No data found for symbol: {symbol}"
            
            # Standardize column names
            df = df.rename(columns={
                "Stock Splits": "Stock_Splits",
                "Dividends": "Dividends"
            })
            
            # Keep only OHLCV columns
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[required_cols].copy()
            
            # Remove timezone info from index for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Save to cache
            if self.use_cache:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path)
                logger.info(f"Cached data for {yahoo_symbol}")
            
            return df, None
            
        except Exception as e:
            error_msg = f"Error fetching data for {symbol}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def _filter_date_range(
        self, 
        df: pd.DataFrame, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Filter DataFrame to specified date range."""
        mask = (df.index >= start_date) & (df.index <= end_date)
        return df.loc[mask].copy()
    
    def get_latest_price(self, symbol: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Get the latest closing price for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (latest price, error message if any)
        """
        df, error = self.fetch(symbol, years=1)
        
        if error:
            return None, error
            
        if df is None or df.empty:
            return None, f"No data available for {symbol}"
            
        return df["Close"].iloc[-1], None
    
    def get_stock_info(self, symbol: str) -> dict:
        """
        Get basic information about a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        yahoo_symbol = self._normalize_symbol(symbol)
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "yahoo_symbol": yahoo_symbol,
                "name": info.get("longName", info.get("shortName", symbol)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "currency": info.get("currency", "INR"),
            }
        except Exception as e:
            logger.warning(f"Could not fetch info for {symbol}: {e}")
            return {
                "symbol": symbol,
                "yahoo_symbol": yahoo_symbol,
                "name": symbol,
                "sector": "Unknown",
                "industry": "Unknown",
            }


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    fetcher = StockDataFetcher()
    
    # Test fetching data for a popular NSE stock
    df, error = fetcher.fetch("TCS")
    
    if error:
        print(f"Error: {error}")
    else:
        print(f"Fetched {len(df)} days of data for TCS")
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"\nLatest data:\n{df.tail()}")
        
        # Test stock info
        info = fetcher.get_stock_info("TCS")
        print(f"\nStock Info: {info}")
