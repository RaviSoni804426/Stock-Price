
# Model Validation & Reliability Report

## 🚨 Executive Summary
**Status:** **NOT READY FOR LIVE TRADING**
**Confidence:** Low (No consistent alpha found)

We have rigorously audited, repaired, and stress-tested the machine learning pipeline. While the **code is now correct** (no look-ahead bias, correct time-series splitting), the **predictive power is currently insufficient** to generate profitable trade signals.

The system is now "Honest but Weak." It no longer lies about its accuracy, but it correctly identifies that it cannot predict market movements with the current simple feature set.

---

## 📈 New System Logic (Risk-On)

We have transitioned from aimlessly chasing "Accuracy" to a structured **Risk-Taking Model**.

### 1. The Three Zones
Instead of a binary Buy/Sell, the system now outputs three outcomes based on probability:

| Zone | Logic | Action |
| :--- | :--- | :--- |
| **High Conviction** | Prob ≥ 70% | **1.0x** Position Size |
| **Calculated Risk** | Prob 60-70% | **0.5x** Position Size |
| **Avoid** | Prob < 60% | **No Trade** |

### 2. Safety Filters
Even if the probability is high, a trade is **killed** if:
*   **Market Regime**: Nifty 50 is in a downtrend (Price < 50 SMA).
*   **Counter-Trend**: Stock price is below its own 50 SMA.
*   **Extreme Volatility**: Recent volatility is >50% annualized.

### 3. Validation
The `analysis/system_validator.py` script now runs a full simulation of this logic over the past 3 years.
*   **Goal**: Win Rate > 55% in "High Conviction" zone.
*   **Goal**: "Calculated Risk" zone must not drag down the portfolio (Profit Factor > 1.0).

---

## 📂 Artifacts
*   `models/train.py`: Optimized for Precision/Recall logging.
*   `rules/engine.py`: Contains the logic for Risk Zones and Position Sizing.
*   `analysis/system_validator.py`: The new truth-teller simulation script.

**Recommendation**: The system is now technically "Complete". It has a brain (ML), a conscience (Rules), and a memory (Validator). You can now safely experiment with new features knowing the safety guardrails are active.
