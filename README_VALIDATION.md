
# Model Validation & Reliability Report

## 🚨 Executive Summary
**Status:** **NOT READY FOR LIVE TRADING**
**Confidence:** Low (No consistent alpha found)

We have rigorously audited, repaired, and stress-tested the machine learning pipeline. While the **code is now correct** (no look-ahead bias, correct time-series splitting), the **predictive power is currently insufficient** to generate profitable trade signals.

The system is now "Honest but Weak." It no longer lies about its accuracy, but it correctly identifies that it cannot predict market movements with the current simple feature set.

---

## 🛠️ Repairs Implemented
1.  **Look-Ahead Bias Removal**:
    *   *Issue*: The original model churned data randomly, allowing it to "peek" at future prices during training.
    *   *Fix*: Implemented `TimeSeriesSplit`. The model is now strictly trained on past data and tested on future data.
2.  **Target Labeling Correction**:
    *   *Issue*: Fixed percentage thresholds (e.g., gain > 3.5%) were biased toward high-volatility periods.
    *   *Fix*: Implemented **Dynamic ATR-based Labeling**. Targets now scale with volatility (e.g., Target = 2.0 * Daily Volatility).
3.  **Class Imbalance**:
    *   *Fix*: Verified positive class ratios are healthy (10-20%) after dynamic labeling, removing the need for extreme oversampling.

---

## 📉 Performance Analysis (7-Day Horizon)

### 1. Honest Metrics (No Cheating)
| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.50 - 0.55** | **Random Chance**. The model has no discriminatory power. |
| **Accuracy** | ~85% | Misleading. It achieves this by predicting "No Trade" 100% of the time. |
| **Precision** | **0.0%** | The model is too afraid to take any trade. |

### 2. Walk-Forward Analysis (Simulation)
We simulated the model's performance over 5 rolling periods (April 2023 - Sept 2025):
*   **Fold 1 (Apr '23 - Oct '23)**: **ROC-AUC 0.68** (Promising start)
*   **Fold 2 (Oct '23 - Apr '24)**: **ROC-AUC 0.53** (Alpha decayed)
*   **Fold 4 (Sep '24 - Mar '25)**: **ROC-AUC 0.35** (Inverse correlation - disastrous)

**Conclusion**: The model's performance is unstable and often worse than random.

---

## 🔮 Next Steps for Improvement (The "Alpha" Quest)
Since the **infrastructure** is now solid, you can focus purely on finding better features. You do not need to worry about code bugs, only data quality.

### 1. Add "External" Features (Critical)
The current model only looks at *internal* price history (SMA, RSI). It needs external context:
*   **Nifty 50 Index**: Is the market bullish or bearish?
*   **Sector Index**: Is the Bank/IT sector moving?
*   **VIX**: Is implied volatility high?

### 2. Implement "Regime Filters"
Don't trade all the time. Add a rule:
*   "Only trade if Price > SMA(200)" (Trend Filter)
*   "Only trade if ADX > 25" (Momentum Filter)

### 3. Try Different Models
*   **Random Forest**: Might handle noise better than XGBoost.
*   **LSTM/GRU**: Deep learning for sequence patterns (advanced).

---

## 📂 Artifacts
*   `models/train.py`: The clean, safe training script.
*   `features/engineer.py`: The feature engineering with dynamic labeling.
*   `config.py`: Configuration with tunable ATR multipliers.

**Recommendation**: Use this codebase as a **Research Gym**. Do not deploy real capital until ROC-AUC consistently stays above **0.55** in Walk-Forward Validation.
