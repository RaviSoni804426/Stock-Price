"""
Data validation module.
Validates stock data for quality and suitability for ML training/prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging

import sys
sys.path.insert(0, str(__file__).rsplit("\\", 2)[0])
from config import (
    MIN_TRADING_DAYS, 
    VALIDATION_CONFIG
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: dict
    
    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        msg = f"Validation: {status}"
        if self.errors:
            msg += f"\n  Errors: {', '.join(self.errors)}"
        if self.warnings:
            msg += f"\n  Warnings: {', '.join(self.warnings)}"
        return msg


class DataValidator:
    """
    Validates stock data for quality and ML suitability.
    
    Checks for:
    - Sufficient history
    - Liquidity (volume)
    - Price reasonability
    - Data gaps and missing values
    - Abnormal price movements
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize validator with configuration.
        
        Args:
            config: Validation thresholds (uses defaults if not provided)
        """
        self.config = config or VALIDATION_CONFIG
        
    def validate(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> ValidationResult:
        """
        Run all validation checks on the data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol for logging
            
        Returns:
            ValidationResult with status, errors, warnings, and stats
        """
        errors = []
        warnings = []
        stats = {}
        
        if df is None or df.empty:
            return ValidationResult(
                is_valid=False,
                errors=["No data provided"],
                warnings=[],
                stats={}
            )
        
        # Check 1: Sufficient history
        history_check, history_msg, history_stats = self._check_history(df)
        stats.update(history_stats)
        if not history_check:
            errors.append(history_msg)
        
        # Check 2: Liquidity
        liquidity_check, liquidity_msg, liquidity_stats = self._check_liquidity(df)
        stats.update(liquidity_stats)
        if not liquidity_check:
            errors.append(liquidity_msg)
        elif "low" in liquidity_msg.lower():
            warnings.append(liquidity_msg)
        
        # Check 3: Price validity
        price_check, price_msg, price_stats = self._check_price(df)
        stats.update(price_stats)
        if not price_check:
            errors.append(price_msg)
        
        # Check 4: Missing data
        missing_check, missing_msg, missing_stats = self._check_missing_data(df)
        stats.update(missing_stats)
        if not missing_check:
            errors.append(missing_msg)
        elif missing_stats.get("missing_pct", 0) > 1:
            warnings.append(missing_msg)
        
        # Check 5: Abnormal gaps
        gap_check, gap_msg, gap_stats = self._check_price_gaps(df)
        stats.update(gap_stats)
        if not gap_check:
            warnings.append(gap_msg)  # Gaps are warnings, not errors
        
        # Check 6: Data consistency
        consistency_check, consistency_msg = self._check_consistency(df)
        if not consistency_check:
            errors.append(consistency_msg)
        
        is_valid = len(errors) == 0
        
        logger.info(f"Validation for {symbol}: {'PASSED' if is_valid else 'FAILED'}")
        if errors:
            logger.warning(f"Validation errors for {symbol}: {errors}")
        if warnings:
            logger.info(f"Validation warnings for {symbol}: {warnings}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    def _check_history(self, df: pd.DataFrame) -> Tuple[bool, str, dict]:
        """Check if data has sufficient history."""
        trading_days = len(df)
        years_approx = trading_days / 252  # Approximate trading days per year
        
        stats = {
            "trading_days": trading_days,
            "years_approx": round(years_approx, 2),
            "start_date": df.index.min().strftime("%Y-%m-%d"),
            "end_date": df.index.max().strftime("%Y-%m-%d"),
        }
        
        if trading_days < MIN_TRADING_DAYS:
            return False, f"Insufficient history: {trading_days} days (need {MIN_TRADING_DAYS})", stats
        
        return True, f"History OK: {trading_days} days (~{years_approx:.1f} years)", stats
    
    def _check_liquidity(self, df: pd.DataFrame) -> Tuple[bool, str, dict]:
        """Check if stock has sufficient liquidity."""
        avg_volume = df["Volume"].mean()
        recent_avg_volume = df["Volume"].tail(60).mean()  # Last ~3 months
        
        stats = {
            "avg_volume": int(avg_volume),
            "recent_avg_volume": int(recent_avg_volume),
        }
        
        min_volume = self.config["min_avg_volume"]
        
        if avg_volume < min_volume / 2:
            return False, f"Very illiquid: avg volume {avg_volume:,.0f} (need {min_volume:,})", stats
        
        if avg_volume < min_volume:
            return True, f"Low liquidity warning: avg volume {avg_volume:,.0f}", stats
        
        return True, f"Liquidity OK: avg volume {avg_volume:,.0f}", stats
    
    def _check_price(self, df: pd.DataFrame) -> Tuple[bool, str, dict]:
        """Check if price is within reasonable bounds."""
        latest_price = df["Close"].iloc[-1]
        min_price = df["Close"].min()
        max_price = df["Close"].max()
        
        stats = {
            "latest_price": round(latest_price, 2),
            "min_price": round(min_price, 2),
            "max_price": round(max_price, 2),
        }
        
        min_allowed = self.config["min_price"]
        
        if latest_price < min_allowed:
            return False, f"Price too low: ₹{latest_price:.2f} (min ₹{min_allowed})", stats
        
        if min_price <= 0:
            return False, "Invalid price data: contains zero or negative prices", stats
        
        return True, f"Price OK: ₹{latest_price:.2f}", stats
    
    def _check_missing_data(self, df: pd.DataFrame) -> Tuple[bool, str, dict]:
        """Check for missing values in the data."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100
        
        # Check for NaN in critical columns
        critical_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_per_col = {col: df[col].isnull().sum() for col in critical_cols if col in df.columns}
        
        stats = {
            "missing_cells": missing_cells,
            "missing_pct": round(missing_pct, 2),
            "missing_per_column": missing_per_col,
        }
        
        max_missing = self.config["max_missing_pct"]
        
        if missing_pct > max_missing:
            return False, f"Too much missing data: {missing_pct:.1f}% (max {max_missing}%)", stats
        
        if missing_pct > 1:
            return True, f"Some missing data: {missing_pct:.1f}%", stats
        
        return True, f"Missing data OK: {missing_pct:.2f}%", stats
    
    def _check_price_gaps(self, df: pd.DataFrame) -> Tuple[bool, str, dict]:
        """Check for abnormal price gaps."""
        # Calculate daily returns
        returns = df["Close"].pct_change() * 100
        
        max_gap = returns.abs().max()
        large_gaps = (returns.abs() > self.config["max_gap_pct"]).sum()
        
        stats = {
            "max_daily_gap_pct": round(max_gap, 2),
            "large_gap_count": int(large_gaps),
        }
        
        if large_gaps > 10:
            return False, f"Many large gaps: {large_gaps} days with >{self.config['max_gap_pct']}% moves", stats
        
        if large_gaps > 0:
            return True, f"Some large gaps: {large_gaps} days with >{self.config['max_gap_pct']}% moves", stats
        
        return True, "No abnormal gaps", stats
    
    def _check_consistency(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check data consistency (High >= Low, etc.)."""
        # High should be >= Low
        invalid_hl = (df["High"] < df["Low"]).sum()
        
        # High should be >= Open and Close
        invalid_high = ((df["High"] < df["Open"]) | (df["High"] < df["Close"])).sum()
        
        # Low should be <= Open and Close
        invalid_low = ((df["Low"] > df["Open"]) | (df["Low"] > df["Close"])).sum()
        
        total_invalid = invalid_hl + invalid_high + invalid_low
        
        if total_invalid > 0:
            return False, f"Inconsistent OHLC data: {total_invalid} invalid rows"
        
        return True, "Data consistency OK"
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Forward fill missing values (common in stock data)
        df = df.ffill()
        
        # Drop any remaining NaN rows at the start
        df = df.dropna()
        
        # Remove rows with zero volume (usually holidays/errors)
        df = df[df["Volume"] > 0]
        
        # Remove duplicate indices
        df = df[~df.index.duplicated(keep="first")]
        
        # Sort by date
        df = df.sort_index()
        
        return df


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Import fetcher and test validation
    from fetcher import StockDataFetcher
    
    fetcher = StockDataFetcher()
    validator = DataValidator()
    
    # Test with a stock
    df, error = fetcher.fetch("RELIANCE")
    
    if error:
        print(f"Fetch error: {error}")
    else:
        result = validator.validate(df, "RELIANCE")
        print(result)
        print(f"\nStats: {result.stats}")
        
        # Clean the data
        df_clean = validator.clean_data(df)
        print(f"\nOriginal rows: {len(df)}, After cleaning: {len(df_clean)}")
