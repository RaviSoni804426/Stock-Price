
"""
System Validator (Task 6)
Performs Walk-Forward Validation on the FULL DECISION SYSTEM (Model + Rules).
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import StockDataFetcher
from data.validator import DataValidator
from features.engineer import FeatureEngineer
from models.train import train_model
from rules.engine import RulesEngine, Decision

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("SystemValidator")

class SystemValidator:
    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.validator = DataValidator()
        self.engineer = FeatureEngineer()
        self.rules = RulesEngine()
        
    def fetch_market_data(self):
        """Fetch Nifty data for the whole period."""
        logger.info("Fetching Nifty data for Market Regime...")
        df, _ = self.fetcher.fetch("NIFTY", years=10)
        # Compute 50-day SMA for trend
        df["sma_50"] = df["Close"].rolling(50).mean()
        df["market_trend"] = np.where(df["Close"] > df["sma_50"], "bullish", "bearish")
        return df
    
    def validate_walk_forward(self, start_year=2022, end_year=2024, stocks=None):
        if stocks is None:
            # Use a subset of diverse stocks for speed
            stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "TATAMOTORS", "SBIN", "ITC"]
            
        logger.info(f"Starting Walk-Forward Validation on {len(stocks)} stocks...")
        
        # 1. Fetch all stock data
        stock_data = {}
        for s in stocks:
            df, _ = self.fetcher.fetch(s, years=7) # Enough history
            if df is not None:
                # Feature Engineering
                df = self.engineer.compute_features(df)
                df["future_return"] = df["Close"].shift(-7).pct_change(periods=7) * 100 # 7-day future return
                stock_data[s] = df
        
        market_df = self.fetch_market_data()
        
        all_trades = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"\n{'='*40}")
            logger.info(f"PROCESSING YEAR: {year}")
            logger.info(f"{'='*40}")
            
            # 2. Train on history ( < year )
            train_X_list = []
            train_y_list = []
            
            cutoff_date = datetime(year, 1, 1)
            
            for s, df in stock_data.items():
                # Filter training data
                train_mask = df.index < cutoff_date
                train_df = df[train_mask].copy()
                
                # Re-create labels for 7-day horizon
                labeled_df = self.engineer.create_labels(train_df, horizon=7)
                labeled_df = labeled_df.dropna(subset=self.engineer.FEATURE_COLUMNS + ["target"])
                
                if not labeled_df.empty:
                    train_X_list.append(labeled_df[self.engineer.FEATURE_COLUMNS])
                    train_y_list.append(labeled_df["target"])
            
            if not train_X_list:
                logger.warning(f"No training data for {year}")
                continue
                
            X_train = pd.concat(train_X_list)
            y_train = pd.concat(train_y_list)
            
            logger.info(f"Training Model for {year} (Samples: {len(X_train)})...")
            model, _ = train_model(X_train, y_train, horizon=7)
            
            # 3. Simulate Trading Year
            logger.info(f"Simulating trades for {year}...")
            
            total_trades = 0
            
            for s, df in stock_data.items():
                year_mask = (df.index.year == year)
                test_df = df[year_mask].copy()
                
                if test_df.empty: continue
                
                # Predict
                X_test = test_df[self.engineer.FEATURE_COLUMNS]
                if X_test.empty: continue
                
                probs = model.predict_proba(X_test)[:, 1]
                
                for i in range(len(test_df)):
                    date = test_df.index[i]
                    prob = probs[i]
                    price = test_df["Close"].iloc[i]
                    
                    # Prepare ML Output dict for Rules Engine
                    ml_output = {
                        "probability": prob,
                        "trend": "bullish" if prob > 0.5 else "bearish", 
                        "expected_move": 0.0,
                        "features": {
                            "rsi": test_df["rsi_14"].iloc[i],
                            "above_sma_50": test_df["above_sma_50"].iloc[i],
                            "volatility": test_df["volatility_20d"].iloc[i]
                        }
                    }
                    
                    # Get Market Context
                    if date in market_df.index:
                        m_trend = market_df.loc[date, "market_trend"]
                    else:
                        m_trend = "neutral"
                        
                    # Evaluate
                    plan = self.rules.evaluate(
                        ml_output=ml_output,
                        horizon=7,
                        latest_price=price,
                        market_trend=m_trend
                    )
                    
                    if plan.decision == Decision.BUY:
                        entry = price
                        try:
                            # 7 trading days later (approx)
                            idx_loc = df.index.get_loc(date)
                            if idx_loc + 7 >= len(df): continue
                            
                            exit_price = df["Close"].iloc[idx_loc + 7]
                            pct_return = ((exit_price - entry) / entry) * 100
                            
                            all_trades.append({
                                "date": date,
                                "symbol": s,
                                "zone": "HIGH_CONVICTION" if plan.confidence >= 75 else "CALCULATED_RISK",
                                "prob": prob,
                                "return_pct": pct_return,
                                "position_size": plan.position_size_multiplier,
                                "market_trend": m_trend
                            })
                            total_trades += 1
                        except IndexError:
                            pass 
            
            logger.info(f"Year {year} finished. Trades found: {total_trades}")

        # 4. Report
        trades_df = pd.DataFrame(all_trades)
        if trades_df.empty:
            logger.warning("No trades generated across all years.")
            return
            
        logger.info("\n" + "="*50)
        logger.info("FINAL VALIDATION REPORT (Task 6)")
        logger.info("="*50)
        
        for zone in ["HIGH_CONVICTION", "CALCULATED_RISK"]:
            subset = trades_df[trades_df["zone"] == zone]
            if subset.empty:
                logger.info(f"Zone {zone}: No trades.")
                continue
                
            win_rate = (subset["return_pct"] > 0).mean() * 100
            avg_return = subset["return_pct"].mean()
            median_return = subset["return_pct"].median()
            count = len(subset)
            
            # Simple Sharpe Proxy
            sharpe = avg_return / subset["return_pct"].std() if len(subset) > 1 else 0
            
            logger.info(f"\nZone: {zone}")
            logger.info(f"  Count:      {count}")
            logger.info(f"  Win Rate:   {win_rate:.2f}% (Target: 55-60%)")
            logger.info(f"  Avg Return: {avg_return:.2f}%")
            logger.info(f"  Median Ret: {median_return:.2f}%")
            logger.info(f"  Sharpe:     {sharpe:.2f}")
            
            if zone == "CALCULATED_RISK" and win_rate < 50:
                 logger.warning("  FAILURE: Calculated Risk Buyer Precision < 50%")
            
        logger.info("="*50)

if __name__ == "__main__":
    validator = SystemValidator()
    validator.validate_walk_forward()
