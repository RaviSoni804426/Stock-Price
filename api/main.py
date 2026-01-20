"""
FastAPI Backend for Stock Decision Assistant.

Endpoints:
- POST /generate-plan: Generate investment plan for a stock
- GET /health: Health check
- GET /available-horizons: Get list of available prediction horizons

Usage:
    python -m api.main
    OR
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import logging
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import HORIZONS, API_HOST, API_PORT
from data.fetcher import StockDataFetcher
from data.validator import DataValidator
from models.predictor import StockPredictor
from rules.engine import RulesEngine
from llm.explainer import PlanExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Decision Assistant API",
    description="AI-powered stock decision assistant for NSE stocks",
    version="1.0.0",
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (lazy loading for predictor)
fetcher = StockDataFetcher()
data_validator = DataValidator()
rules_engine = RulesEngine()
explainer = PlanExplainer()
predictor: Optional[StockPredictor] = None


def get_predictor() -> StockPredictor:
    """Lazy load the predictor to avoid loading models on import."""
    global predictor
    if predictor is None:
        predictor = StockPredictor()
    return predictor


# Request/Response Models
class PlanRequest(BaseModel):
    """Request model for generate-plan endpoint."""
    stock: str = Field(..., description="NSE stock symbol (e.g., INFOSYS, TCS)")
    days: int = Field(..., ge=3, le=15, description="Holding period in days (3-15)")
    
    @field_validator("stock")
    @classmethod
    def validate_stock(cls, v):
        """Validate and normalize stock symbol."""
        return v.upper().strip()
    
    @field_validator("days")
    @classmethod
    def validate_days(cls, v):
        """Validate holding period is in allowed horizons."""
        if v not in HORIZONS:
            # Find closest available horizon
            closest = min(HORIZONS, key=lambda x: abs(x - v))
            logger.warning(f"Requested {v} days not available, using closest: {closest}")
            return closest
        return v


class PlanResponse(BaseModel):
    """Response model for generate-plan endpoint."""
    decision: str = Field(..., description="BUY or AVOID")
    confidence: int = Field(..., ge=0, le=100, description="Confidence percentage")
    entry: float = Field(..., description="Entry price in INR")
    stop_loss: float = Field(..., description="Stop-loss price in INR")
    target: float = Field(..., description="Target price in INR")
    explanation: str = Field(..., description="Human-readable explanation")
    
    # Additional fields
    stock: str
    days: int
    risk_reward_ratio: float
    position_risk_pct: float
    ml_probability: float
    ml_trend: str
    timestamp: str
    
    # Warnings
    warnings: List[str] = []


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: int
    available_horizons: List[int]
    timestamp: str


# Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Stock Decision Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate-plan": "Generate investment plan",
            "GET /health": "Health check",
            "GET /available-horizons": "List available horizons",
        },
        "disclaimer": "This API is for educational purposes only. Not financial advice.",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    pred = get_predictor()
    
    return HealthResponse(
        status="healthy",
        models_loaded=len(pred.get_available_horizons()),
        available_horizons=pred.get_available_horizons(),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/available-horizons", response_model=List[int])
async def get_available_horizons():
    """Get list of available prediction horizons."""
    pred = get_predictor()
    return pred.get_available_horizons()


@app.post("/generate-plan", response_model=PlanResponse, responses={400: {"model": ErrorResponse}})
async def generate_plan(request: PlanRequest):
    """
    Generate an investment plan for a stock.
    
    Takes a stock symbol and holding period, returns a complete investment plan
    with entry, stop-loss, target prices and an explanation.
    """
    logger.info(f"Received request: {request.stock} for {request.days} days")
    
    pred = get_predictor()
    
    # Check if model is available for requested horizon
    if not pred.is_model_available(request.days):
        available = pred.get_available_horizons()
        if not available:
            raise HTTPException(
                status_code=503,
                detail="No models available. Please train models first using: python -m models.train"
            )
        # Find closest available horizon
        request.days = min(available, key=lambda x: abs(x - request.days))
        logger.warning(f"Using closest available horizon: {request.days} days")
    
    try:
        # Step 1: Fetch data
        df, error = fetcher.fetch(request.stock)
        if error:
            raise HTTPException(status_code=400, detail=f"Data fetch failed: {error}")
        
        # Step 2: Validate data
        validation_result = data_validator.validate(df, request.stock)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400, 
                detail=f"Data validation failed: {', '.join(validation_result.errors)}"
            )
        
        # Step 3: Clean data
        df = data_validator.clean_data(df)
        
        # Step 4: Get ML prediction
        ml_output = pred.predict(df, request.days)
        
        # Step 5: Apply rules engine
        plan = rules_engine.evaluate(
            ml_output=ml_output,
            horizon=request.days,
            latest_price=ml_output["latest_close"]
        )
        
        # Step 6: Generate explanation
        stock_info = fetcher.get_stock_info(request.stock)
        explanation = explainer.explain(
            plan=plan,
            stock_symbol=request.stock,
            horizon=request.days,
            stock_info=stock_info
        )
        
        # Build response
        response = PlanResponse(
            decision=plan.decision,
            confidence=plan.confidence,
            entry=plan.entry_price,
            stop_loss=plan.stop_loss,
            target=plan.target_price,
            explanation=explanation,
            stock=request.stock,
            days=request.days,
            risk_reward_ratio=plan.risk_reward_ratio,
            position_risk_pct=plan.position_risk_pct,
            ml_probability=plan.ml_probability,
            ml_trend=plan.ml_trend,
            timestamp=datetime.now().isoformat(),
            warnings=plan.warnings,
        )
        
        logger.info(f"Generated plan for {request.stock}: {plan.decision} with {plan.confidence}% confidence")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating plan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/stock-info/{symbol}")
async def get_stock_info(symbol: str):
    """Get basic information about a stock."""
    symbol = symbol.upper().strip()
    info = fetcher.get_stock_info(symbol)
    return info


@app.get("/validate/{symbol}")
async def validate_stock(symbol: str):
    """Validate if a stock has sufficient data for analysis."""
    symbol = symbol.upper().strip()
    
    df, error = fetcher.fetch(symbol)
    if error:
        return {"valid": False, "error": error}
    
    result = data_validator.validate(df, symbol)
    
    return {
        "valid": result.is_valid,
        "symbol": symbol,
        "errors": result.errors,
        "warnings": result.warnings,
        "stats": result.stats,
    }


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    pred = get_predictor()
    available = pred.get_available_horizons()
    
    if not available:
        print("\n" + "="*60)
        print("WARNING: No trained models found!")
        print("Please train models first using:")
        print("  python -m models.train")
        print("="*60 + "\n")
    else:
        print(f"\nModels loaded for horizons: {available}")
    
    print(f"\nStarting server at http://{API_HOST}:{API_PORT}")
    print("API documentation available at http://localhost:8000/docs\n")
    
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
