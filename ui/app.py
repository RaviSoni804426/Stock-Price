"""
Streamlit UI for Stock Decision Assistant.

A clean, professional UI for generating stock investment plans.

Usage:
    streamlit run ui/app.py
"""

import streamlit as st
import requests
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Stock Decision Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Decision badges */
    .decision-buy {
        background-color: #10B981;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 1.25rem;
        display: inline-block;
    }
    
    .decision-avoid {
        background-color: #EF4444;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 1.25rem;
        display: inline-block;
    }
    
    /* Price cards */
    .price-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .price-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .price-card-red {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
    }
    
    .price-label {
        font-size: 0.875rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .price-value {
        font-size: 1.75rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    
    /* Confidence bar */
    .confidence-container {
        margin: 1rem 0;
    }
    
    /* Disclaimer */
    .disclaimer {
        background-color: red;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 2rem;
    }
    
    /* Header */
    .header-container {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API is running and has models loaded."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_available_horizons():
    """Get available prediction horizons from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/available-horizons", timeout=5)
        if response.status_code == 200:
            return response.json()
        return [3, 5, 7, 10, 15]
    except:
        return [3, 5, 7, 10, 15]


def generate_plan(stock: str, days: int):
    """Call API to generate investment plan."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-plan",
            json={"stock": stock, "days": days},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            error_data = response.json()
            return None, error_data.get("detail", "Unknown error")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Please ensure the server is running."
    except Exception as e:
        return None, str(e)


def render_header():
    """Render the page header."""
    st.markdown("""
    <div class="header-container">
        <h1 style="margin:0; font-size: 2.5rem;">📈 Stock Decision Assistant</h1>
        <p style="margin-top: 0.5rem; opacity: 0.8;">
            AI-powered investment analysis for NSE stocks
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_decision_badge(decision: str, confidence: int):
    """Render the decision badge."""
    badge_class = "decision-buy" if decision == "BUY" else "decision-avoid"
    st.markdown(f"""
    <div style="text-align: center; margin: 1.5rem 0;">
        <span class="{badge_class}">{decision}</span>
        <p style="margin-top: 0.5rem; color: #6B7280;">
            Confidence: <strong>{confidence}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_price_cards(entry: float, stop_loss: float, target: float):
    """Render price level cards."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">Entry Price</div>
            <div class="price-value">₹{entry:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="price-card price-card-red">
            <div class="price-label">Stop Loss</div>
            <div class="price-value">₹{stop_loss:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="price-card price-card-green">
            <div class="price-label">Target</div>
            <div class="price-value">₹{target:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)


def render_metrics(result: dict):
    """Render additional metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Risk/Reward", f"{result['risk_reward_ratio']}:1")
    
    with col2:
        st.metric("Position Risk", f"{result['position_risk_pct']}%")
    
    with col3:
        st.metric("ML Probability", f"{result['ml_probability']*100:.1f}%")
    
    with col4:
        trend_emoji = "🟢" if result['ml_trend'] == "bullish" else "🔴" if result['ml_trend'] == "bearish" else "🟡"
        st.metric("Trend", f"{trend_emoji} {result['ml_trend'].title()}")


def render_warnings(warnings: list):
    """Render warning messages."""
    if warnings:
        with st.expander("⚠️ Warnings & Notes", expanded=False):
            for warning in warnings:
                st.warning(warning)


def render_disclaimer():
    """Render the disclaimer."""
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Important Disclaimer</strong><br>
        This tool is for <strong>educational and informational purposes only</strong>. 
        It does not constitute financial advice. Past performance does not guarantee future results. 
        Always conduct your own research and consult a qualified financial advisor before making investment decisions.
        The creators of this tool are not responsible for any financial losses.
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Check API health
        health = check_api_health()
        if health:
            st.success(f"✅ API Connected")
            st.info(f"Models loaded: {health['models_loaded']}")
        else:
            st.error("❌ API Not Connected")
            st.warning("Start the API server:")
            st.code("python -m api.main", language="bash")
            st.stop()
        
        st.divider()
        
        # Stock input
        st.subheader("📊 Stock Selection")
        
        # Popular stocks dropdown
        popular_stocks = [
            "Select a stock...",
            "TCS", "INFY", "RELIANCE", "HDFCBANK", "ICICIBANK",
            "SBIN", "WIPRO", "HCLTECH", "ITC", "BHARTIARTL",
            "HINDUNILVR", "MARUTI", "TATAMOTORS", "BAJFINANCE", "KOTAKBANK"
        ]
        
        selected_stock = st.selectbox(
            "Popular NSE Stocks",
            popular_stocks,
            help="Select from popular NSE stocks or enter custom symbol below"
        )
        
        custom_stock = st.text_input(
            "Or enter custom symbol",
            placeholder="e.g., ASIANPAINT",
            help="Enter any valid NSE stock symbol"
        )
        
        # Determine which stock to use
        stock_symbol = custom_stock.upper() if custom_stock else (
            selected_stock if selected_stock != "Select a stock..." else ""
        )
        
        st.divider()
        
        # Holding period
        st.subheader("📅 Holding Period")
        available_horizons = get_available_horizons()
        
        days = st.slider(
            "Days to Hold",
            min_value=min(available_horizons),
            max_value=max(available_horizons),
            value=7,
            help="Select your planned holding period (3-15 days)"
        )
        
        # Snap to available horizons
        closest_horizon = min(available_horizons, key=lambda x: abs(x - days))
        if days != closest_horizon:
            st.caption(f"Using closest available: {closest_horizon} days")
            days = closest_horizon
        
        st.divider()
        
        # Generate button
        generate_button = st.button(
            "🚀 Generate Plan",
            type="primary",
            use_container_width=True,
            disabled=not stock_symbol
        )
    
    # Main content
    if not stock_symbol:
        # Show welcome message
        st.markdown("""
        ### 👋 Welcome!
        
        Use the sidebar to:
        1. Select a stock from the dropdown or enter a custom NSE symbol
        2. Choose your holding period (3-15 days)
        3. Click **Generate Plan** to get your analysis
        
        ---
        
        #### How it works:
        
        1. **Data Collection**: Fetches 5+ years of historical data from Yahoo Finance
        2. **Feature Engineering**: Computes technical indicators (SMA, RSI, volatility, etc.)
        3. **ML Prediction**: XGBoost model predicts probability of positive movement
        4. **Rules Engine**: Applies risk rules to determine BUY/AVOID decision
        5. **Explanation**: Generates human-readable investment plan
        
        #### Example Stocks:
        - **IT Sector**: TCS, INFY, WIPRO, HCLTECH
        - **Banking**: HDFCBANK, ICICIBANK, SBIN, KOTAKBANK
        - **Others**: RELIANCE, ITC, BHARTIARTL, MARUTI
        """)
        
        render_disclaimer()
        return
    
    # Generate plan when button is clicked or show previous result
    if generate_button:
        with st.spinner(f"Analyzing {stock_symbol}..."):
            result, error = generate_plan(stock_symbol, days)
        
        if error:
            st.error(f"❌ Error: {error}")
            return
        
        # Store result in session state
        st.session_state['last_result'] = result
        st.session_state['last_stock'] = stock_symbol
        st.session_state['last_days'] = days
    
    # Display results if available
    if 'last_result' in st.session_state:
        result = st.session_state['last_result']
        
        st.subheader(f"Analysis for {result['stock']} ({result['days']}-day hold)")
        st.caption(f"Generated at: {result['timestamp']}")
        
        # Decision badge
        render_decision_badge(result['decision'], result['confidence'])
        
        # Confidence bar
        st.progress(result['confidence'] / 100)
        
        # Price cards
        render_price_cards(result['entry'], result['stop_loss'], result['target'])
        
        # Additional metrics
        st.divider()
        render_metrics(result)
        
        # Explanation
        st.divider()
        st.subheader("📝 Analysis Details")
        
        with st.expander("Full Explanation", expanded=True):
            st.markdown(result['explanation'])
        
        # Warnings
        render_warnings(result['warnings'])
        
        # Export option
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            json_str = json.dumps(result, indent=2)
            st.download_button(
                "📥 Download JSON",
                json_str,
                file_name=f"{result['stock']}_{result['days']}d_plan.json",
                mime="application/json"
            )
        
        with col2:
            # Summary text
            summary = f"""
Stock Decision Assistant - Analysis Report
==========================================
Stock: {result['stock']}
Holding Period: {result['days']} days
Generated: {result['timestamp']}

DECISION: {result['decision']}
Confidence: {result['confidence']}%

Price Levels:
- Entry: ₹{result['entry']:,.2f}
- Stop-Loss: ₹{result['stop_loss']:,.2f}
- Target: ₹{result['target']:,.2f}

Metrics:
- Risk/Reward: {result['risk_reward_ratio']}:1
- Position Risk: {result['position_risk_pct']}%
- ML Probability: {result['ml_probability']*100:.1f}%
- Trend: {result['ml_trend']}

DISCLAIMER: This is for educational purposes only. Not financial advice.
"""
            st.download_button(
                "📄 Download Summary",
                summary,
                file_name=f"{result['stock']}_{result['days']}d_summary.txt",
                mime="text/plain"
            )
    
    render_disclaimer()


if __name__ == "__main__":
    main()
