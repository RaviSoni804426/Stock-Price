
"""
Stock Decision Assistant - Hugging Face Deployment
A monolithic Streamlit app that runs the full ML pipeline locally.
"""

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import HORIZONS
from data.fetcher import StockDataFetcher
from data.validator import DataValidator
from models.predictor import StockPredictor
from rules.engine import RulesEngine
from llm.explainer import PlanExplainer

# Configure Logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Decision Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHED RESOURCES (Singleton Logic) ---
@st.cache_resource
def load_components():
    """Load and cache the backend components."""
    return {
        "fetcher": StockDataFetcher(),
        "validator": DataValidator(),
        "predictor": StockPredictor(),
        "rules": RulesEngine(),
        "explainer": PlanExplainer()
    }

components = load_components()

# --- DIRECT LOGIC (Replaces API) ---

def check_models_loaded():
    """Check which models are loaded."""
    try:
        available = components["predictor"].get_available_horizons()
        return available
    except Exception as e:
        return []

def generate_plan_local(stock_symbol: str, days: int):
    """Run the full analysis pipeline locally (no API call)."""
    try:
        # 1. Fetch
        df, error = components["fetcher"].fetch(stock_symbol)
        if error:
            return None, f"Data fetch failed: {error}"
            
        # 2. Validate
        val_res = components["validator"].validate(df, stock_symbol)
        if not val_res.is_valid:
            return None, f"Data validation failed: {', '.join(val_res.errors)}"
            
        # 3. Clean
        df = components["validator"].clean_data(df)
        
        # 4a. Market Context (Task 4)
        # Fetch Nifty data to determine market regime
        market_trend = "neutral"
        try:
            nifty_df, n_err = components["fetcher"].fetch("NIFTY", years=1) # "NIFTY" maps to ^NSEI
            if not n_err and not nifty_df.empty:
                 # Simple Trend: Price > SMA 50
                 latest_nifty = nifty_df["Close"].iloc[-1]
                 nifty_sma50 = nifty_df["Close"].rolling(window=50).mean().iloc[-1]
                 
                 if latest_nifty > nifty_sma50:
                     market_trend = "bullish"
                 else:
                     market_trend = "bearish"
        except Exception as e:
            logger.warning(f"Failed to fetch Market Context: {e}")

        
        # 4b. Predict
        # Check model availability first
        pred = components["predictor"]
        if not pred.is_model_available(days):
            available = pred.get_available_horizons()
            if not available:
                return None, "No trained models found. Please train models locally and upload the .pkl files."
            
            # Snap to closest
            closest = min(available, key=lambda x: abs(x - days))
            days = closest # Update days to use closest
        
        ml_output = pred.predict(df, days)
        
        # 5. Rules
        plan = components["rules"].evaluate(
            ml_output=ml_output,
            horizon=days,
            latest_price=ml_output["latest_close"],
            market_trend=market_trend
        )
        
        # 6. Explain
        stock_info = components["fetcher"].get_stock_info(stock_symbol)
        explanation = components["explainer"].explain(
            plan=plan,
            stock_symbol=stock_symbol,
            horizon=days,
            stock_info=stock_info
        )
        
        # 7. Format Result
        return {
            "decision": plan.decision,
            "confidence": plan.confidence,
            "entry": plan.entry_price,
            "stop_loss": plan.stop_loss,
            "target": plan.target_price,
            "explanation": explanation,
            "stock": stock_symbol,
            "days": days,
            "risk_reward_ratio": plan.risk_reward_ratio,
            "position_risk_pct": plan.position_risk_pct,
            "ml_probability": plan.ml_probability,
            "ml_trend": plan.ml_trend,
            "timestamp": datetime.now().isoformat(),
            "warnings": plan.warnings,
            "market_trend": market_trend,
            "position_size": plan.position_size_multiplier,
        }, None
        
    except Exception as e:
        return None, f"Analysis Error: {str(e)}"

# --- UI COMPONENTS ---

st.markdown("""
<style>
    /* Decision badges */
    .decision-buy { background-color: #10B981; color: white; padding: 0.5rem 1.5rem; border-radius: 99px; font-weight: bold; font-size: 1.25rem; }
    .decision-avoid { background-color: #EF4444; color: white; padding: 0.5rem 1.5rem; border-radius: 99px; font-weight: bold; font-size: 1.25rem; }
    /* Price cards */
    .price-card { background: #1F2937; padding: 1rem; border-radius: 12px; color: white; text-align: center; border: 1px solid #374151; }
    .price-card-green { border-bottom: 4px solid #10B981; }
    .price-card-red { border-bottom: 4px solid #EF4444; }
    .price-value { font-size: 1.5rem; font-weight: bold; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

def render_header():
    st.title("📈 Stock Decision Assistant")
    st.markdown("AI-powered swing trading analysis for NSE stocks (Validator / Gym Version)")

def render_metrics(result):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk/Reward", f"{result['risk_reward_ratio']}:1")
    
    # Position Size Metric
    size_mult = result.get('position_size', 1.0)
    size_str = "100%" if size_mult == 1.0 else f"{int(size_mult*100)}%"
    c2.metric("Position Size", size_str, help="Recommended position size based on risk tier")
    
    c3.metric("ML Prob", f"{result['ml_probability']*100:.1f}%")
    
    # Market Trend Metric
    m_trend = result.get('market_trend', 'neutral')
    m_icon = "🟢" if m_trend == 'bullish' else ("🔴" if m_trend == 'bearish' else "⚪")
    c4.metric("Market", f"{m_icon} {m_trend.title()}")

def main():
    render_header()
    
    with st.sidebar:
        st.header("Configuration")
        
        # Model Check
        loaded_models = check_models_loaded()
        if loaded_models:
            st.success(f"Models Ready: {loaded_models}")
        else:
            st.error("No Models Found!")
            st.info("Train models locally first using 'python -m models.train' and upload the 'models/saved' folder.")
        
        # Inputs
        stock_symbol = st.text_input("Stock Symbol (NSE)", value="RELIANCE", help="e.g., TCS, INFY, TATAMOTORS").upper()
        
        days_input = st.slider("Horizon (Days)", 3, 15, 7)
        
        if st.button("🚀 Generate Plan", type="primary"):
            if not stock_symbol:
                st.warning("Enter a stock symbol.")
            else:
                with st.spinner(f"Analyzing {stock_symbol}..."):
                    result, error = generate_plan_local(stock_symbol, days_input)
                
                if error:
                    st.error(error)
                else:
                    st.session_state['result'] = result

    # Display Result
    if 'result' in st.session_state:
        res = st.session_state['result']
        
        st.divider()
        c_main, c_conf = st.columns([2, 1])
        
        with c_main:
            st.subheader(f"{res['stock']} ({res['days']} Days)")
            badge_class = "decision-buy" if res['decision'] == "BUY" else "decision-avoid"
            st.markdown(f'<span class="{badge_class}">{res["decision"]}</span>', unsafe_allow_html=True)
        
        with c_conf:
            st.metric("Confidence", f"{res['confidence']}%")
            st.progress(res['confidence'] / 100)
            
        render_metrics(res)
        
        # Price Cards
        st.write("#### Price Levels")
        p1, p2, p3 = st.columns(3)
        p1.markdown(f"""
        <div class="price-card">
            <div>Entry</div>
            <div class="price-value">₹{res['entry']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        p2.markdown(f"""
        <div class="price-card price-card-red">
            <div>Stop Loss</div>
            <div class="price-value">₹{res['stop_loss']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        p3.markdown(f"""
        <div class="price-card price-card-green">
            <div>Target</div>
            <div class="price-value">₹{res['target']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📝 Full Analysis", expanded=True):
            st.markdown(res['explanation'])
            
        if res['warnings']:
            st.warning("\n".join(res['warnings']))

if __name__ == "__main__":
    main()
