"""
LLM Explanation Layer.
Converts structured trading plans into human-readable explanations.

IMPORTANT CONSTRAINTS:
- LLM does NOT predict prices
- LLM does NOT calculate probabilities
- LLM ONLY converts numeric output into readable text
- Input must be structured JSON only
"""

import os
import logging
from typing import Dict, Optional
from string import Template
import json

# Try to import OpenAI, but make it optional
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import sys
sys.path.insert(0, str(__file__).rsplit("\\", 2)[0])
from rules.engine import TradingPlan

logger = logging.getLogger(__name__)


# Template-based explanation (fallback when LLM not available)
EXPLANATION_TEMPLATE = """
## Investment Plan Summary

**Stock:** ${stock_symbol}
**Holding Period:** ${horizon} days
**Analysis Date:** ${date}

### Recommendation: ${decision}

**Confidence Level:** ${confidence}%

### Price Levels
- **Entry Price:** ₹${entry_price}
- **Stop-Loss:** ₹${stop_loss} (${sl_pct}% risk)
- **Target Price:** ₹${target_price} (${target_pct}% potential gain)
- **Risk-Reward Ratio:** ${risk_reward}:1

### Analysis Summary

${analysis_summary}

### Key Factors

${key_factors}

### Risk Warnings

${warnings}

---
**Disclaimer:** This analysis is for educational purposes only and does not constitute financial advice. 
Past performance does not guarantee future results. Always conduct your own research and consider 
consulting a qualified financial advisor before making investment decisions.
"""


class PlanExplainer:
    """
    Converts trading plans into human-readable explanations.
    
    Uses OpenAI API when available, falls back to template-based
    explanations when API is not configured.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the explainer.
        
        Args:
            api_key: OpenAI API key (reads from OPENAI_API_KEY env var if not provided)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            if not OPENAI_AVAILABLE:
                logger.info("OpenAI package not installed. Using template-based explanations.")
            else:
                logger.info("No OpenAI API key configured. Using template-based explanations.")
    
    def explain(
        self, 
        plan: TradingPlan, 
        stock_symbol: str,
        horizon: int,
        stock_info: Optional[Dict] = None
    ) -> str:
        """
        Generate human-readable explanation for a trading plan.
        
        Args:
            plan: TradingPlan from the rules engine
            stock_symbol: Stock symbol
            horizon: Trading horizon in days
            stock_info: Optional stock information (name, sector, etc.)
            
        Returns:
            Human-readable explanation string
        """
        # Prepare structured input for LLM
        plan_data = self._prepare_plan_data(plan, stock_symbol, horizon, stock_info)
        
        if self.client:
            try:
                return self._explain_with_llm(plan_data)
            except Exception as e:
                logger.warning(f"LLM explanation failed: {e}. Falling back to template.")
                return self._explain_with_template(plan_data)
        else:
            return self._explain_with_template(plan_data)
    
    def _prepare_plan_data(
        self, 
        plan: TradingPlan, 
        stock_symbol: str,
        horizon: int,
        stock_info: Optional[Dict]
    ) -> Dict:
        """Prepare structured data for explanation generation."""
        from datetime import datetime
        
        # Calculate percentages
        sl_pct = round((plan.entry_price - plan.stop_loss) / plan.entry_price * 100, 2)
        target_pct = round((plan.target_price - plan.entry_price) / plan.entry_price * 100, 2)
        
        return {
            "stock_symbol": stock_symbol,
            "stock_name": stock_info.get("name", stock_symbol) if stock_info else stock_symbol,
            "sector": stock_info.get("sector", "Unknown") if stock_info else "Unknown",
            "horizon": horizon,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "decision": plan.decision,
            "confidence": plan.confidence,
            "entry_price": plan.entry_price,
            "stop_loss": plan.stop_loss,
            "target_price": plan.target_price,
            "sl_pct": sl_pct,
            "target_pct": target_pct,
            "risk_reward": plan.risk_reward_ratio,
            "position_risk_pct": plan.position_risk_pct,
            "ml_probability": round(plan.ml_probability * 100, 1),
            "ml_trend": plan.ml_trend,
            "ml_expected_move": plan.ml_expected_move,
            "reasons": plan.reasons,
            "warnings": plan.warnings,
        }
    
    def _explain_with_llm(self, plan_data: Dict) -> str:
        """Generate explanation using OpenAI API."""
        
        system_prompt = """You are a professional financial analyst assistant. Your role is to convert 
structured trading plan data into clear, professional explanations for retail investors.

CRITICAL RULES:
1. You do NOT make predictions or give advice
2. You ONLY explain the analysis that was already done
3. You must include all numerical values provided
4. You must include the disclaimer
5. Use professional, calm language - no hype
6. Be clear about uncertainty and risks

Output should be in markdown format with clear sections."""

        user_prompt = f"""Convert this trading plan data into a human-readable investment plan explanation.

INPUT DATA (JSON):
{json.dumps(plan_data, indent=2)}

Generate a clear, professional explanation that includes:
1. Executive summary (2-3 sentences)
2. Recommendation (BUY/AVOID) with confidence level
3. Price levels (entry, stop-loss, target) with percentages
4. Key analysis factors (from reasons)
5. Risk warnings (from warnings)
6. Standard disclaimer

Use markdown formatting. Be professional and measured in tone."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Low temperature for consistency
            max_tokens=1000,
        )
        
        explanation = response.choices[0].message.content
        logger.info("Generated LLM explanation successfully")
        
        return explanation
    
    def _explain_with_template(self, plan_data: Dict) -> str:
        """Generate explanation using template (fallback)."""
        
        # Generate analysis summary
        if plan_data["decision"] == "BUY":
            analysis_summary = f"""Based on quantitative analysis, {plan_data['stock_symbol']} shows favorable conditions 
for a {plan_data['horizon']}-day holding period. The model indicates a {plan_data['ml_probability']}% probability 
of positive movement with an expected move of {plan_data['ml_expected_move']:.1f}%. The overall trend is 
assessed as {plan_data['ml_trend']}."""
        else:
            analysis_summary = f"""Based on quantitative analysis, {plan_data['stock_symbol']} does not meet the 
criteria for a favorable entry at this time for a {plan_data['horizon']}-day holding period. The model indicates 
a {plan_data['ml_probability']}% probability with an expected move of {plan_data['ml_expected_move']:.1f}%. 
The overall trend is assessed as {plan_data['ml_trend']}."""
        
        # Format key factors
        key_factors = "\n".join([f"- {reason}" for reason in plan_data["reasons"]])
        
        # Format warnings
        warnings = "\n".join([f"⚠️ {warning}" for warning in plan_data["warnings"]])
        
        # Fill template
        template = Template(EXPLANATION_TEMPLATE)
        explanation = template.substitute(
            stock_symbol=plan_data["stock_symbol"],
            horizon=plan_data["horizon"],
            date=plan_data["date"],
            decision=plan_data["decision"],
            confidence=plan_data["confidence"],
            entry_price=plan_data["entry_price"],
            stop_loss=plan_data["stop_loss"],
            target_price=plan_data["target_price"],
            sl_pct=plan_data["sl_pct"],
            target_pct=plan_data["target_pct"],
            risk_reward=plan_data["risk_reward"],
            analysis_summary=analysis_summary,
            key_factors=key_factors,
            warnings=warnings,
        )
        
        logger.info("Generated template-based explanation")
        
        return explanation.strip()
    
    def generate_short_explanation(self, plan: TradingPlan, stock_symbol: str) -> str:
        """
        Generate a brief one-paragraph explanation.
        
        Args:
            plan: TradingPlan from the rules engine
            stock_symbol: Stock symbol
            
        Returns:
            Brief explanation string
        """
        if plan.decision == "BUY":
            return (
                f"{stock_symbol} shows favorable conditions with a {plan.confidence}% confidence level. "
                f"The analysis suggests an entry at ₹{plan.entry_price} with a stop-loss at ₹{plan.stop_loss} "
                f"({plan.position_risk_pct}% risk) and a target of ₹{plan.target_price}. "
                f"The risk-reward ratio is {plan.risk_reward_ratio}:1. "
                f"This is based on quantitative analysis and is not financial advice."
            )
        else:
            return (
                f"{stock_symbol} does not meet favorable entry criteria at this time. "
                f"The confidence level is {plan.confidence}%. Key concerns: {'; '.join(plan.reasons[:2])}. "
                f"Consider waiting for better conditions. This is not financial advice."
            )


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from rules.engine import RulesEngine, TradingPlan
    
    # Create sample trading plan
    engine = RulesEngine()
    
    ml_output = {
        "probability": 0.65,
        "expected_move": 4.0,
        "trend": "bullish",
        "features": {
            "rsi": 55,
            "above_sma_50": True,
            "price_to_sma_50_pct": 2.5,
            "volatility": 25,
        }
    }
    
    plan = engine.evaluate(ml_output, horizon=7, latest_price=1595.0)
    
    # Test explainer
    explainer = PlanExplainer()
    
    explanation = explainer.explain(
        plan, 
        stock_symbol="INFOSYS",
        horizon=7,
        stock_info={"name": "Infosys Limited", "sector": "Information Technology"}
    )
    
    print("\n" + "="*60)
    print("FULL EXPLANATION:")
    print("="*60)
    print(explanation)
    
    print("\n" + "="*60)
    print("SHORT EXPLANATION:")
    print("="*60)
    print(explainer.generate_short_explanation(plan, "INFOSYS"))
