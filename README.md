# Project Nova: Comprehensive Equitable Credit Scoring Engine

An end-to-end, fair, data-driven credit scoring prototype for Grab partners (drivers and merchants). The system demonstrates **three distinct fairness approaches**: Baseline (no mitigation), Equalized Odds (post-processing), and Reweighing (pre-processing). It simulates realistic operational data, trains ML models, and evaluates/mitigates bias to produce equitable "Nova Scores" (scaled 300–850).

## 🎯 Key Features

- **Three Fairness Models**: Compare baseline, equalized odds, and reweighing approaches
- **Comprehensive Evaluation**: Performance + fairness metrics for all models
- **Real Examples**: Actual partner comparisons showing model differences
- **Production Ready**: Complete pipeline with error handling and progress tracking
- **Explainable AI**: Feature importance, calibrated probabilities, and SHAP integration

## 📁 Repository Structure

### Core Components
-   **requirements-minimal.txt**: Python dependencies
-   **src/generate_data.py**: Synthetic data generation with controlled bias injection
-   **src/train_model.py**: Three-model training pipeline with fairness mitigation
-   **run_project.py**: Complete automated pipeline (generates all three models)
-   **scripts/generate_datasets.ps1**: Batch data generation for multiple scenarios
-   **notebooks/01_eda_and_fairness.ipynb**: Interactive analysis and model comparison

### Documentation
-   **README.md**: Quick start guide (this file)
-   **EXPLANATION.md**: Comprehensive technical documentation with examples
-   **PROJECT_NOVA_FLOWCHART.md**: Visual flowcharts and system architecture
-   **DATA_GENERATION_GUIDE.md**: Detailed data generation instructions
-   **INSTALL.md**: Installation troubleshooting

### Generated During Execution

**📊 Data Files:**
-   **data/partners.csv**: Main synthetic dataset (50K records)
-   **data/partners_scores_baseline.csv**: Baseline model outputs
-   **data/partners_scores_fair.csv**: Equalized odds model outputs
-   **data/partners_scores_reweighed.csv**: Reweighing model outputs
-   **data/metadata.json**: Dataset configuration

**🧠 Models:**
-   **models/model_baseline.pkl**: Standard model (no fairness mitigation)
-   **models/model_fair.pkl**: Equalized odds model (post-processing)
-   **models/model_reweighed.pkl**: Reweighed model (pre-processing)

**📋 Reports:**
-   **reports/metrics_*.json**: Performance metrics for all three models
-   **reports/fairness_*.json**: Comprehensive fairness analysis for all models

### To start fresh delete data, models, and reports folder.

## Quickstart

1. Create and activate a virtual environment (PowerShell on Windows):
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   
3. Install dependencies:

    **✅ RECOMMENDED (tested on Windows):**

    ```bash
    pip install -r requirements-minimal.txt
    ```

    **If issues persist**, see INSTALL.md for troubleshooting

4. Run the complete three-model pipeline:
   ```bash
   python run_project.py
   ```
   
   This generates:
   - 🎯 **Baseline model**: Standard ML without fairness constraints
   - ⚖️ **Equalized odds model**: Post-processing for fair decisions (100x more decisions)
   - ⚖️ **Reweighing model**: Pre-processing for different risk assessments
   
## 🔧 Manual Steps (Alternative)

If you prefer to run steps individually:

### 1. Generate Synthetic Dataset

```bash
python src/generate_data.py --n 50000 --seed 42 --out data/partners.csv
```

### 2. Train Individual Models

**Baseline Model (No Fairness Mitigation):**
```bash
python src/train_model.py \\
 --data data/partners.csv \\
 --model_out models/model_baseline.pkl \\
 --metrics_out reports/metrics_baseline.json \\
 --fairness_out reports/fairness_baseline.json \\
 --scores_out data/partners_scores_baseline.csv \\
 --mitigation none
```

**Equalized Odds Model (Post-processing):**
```bash
python src/train_model.py \\
 --data data/partners.csv \\
 --model_out models/model_fair.pkl \\
 --metrics_out reports/metrics_fair.json \\
 --fairness_out reports/fairness_fair.json \\
 --scores_out data/partners_scores_fair.csv \\
 --mitigation equalized_odds
```

**Reweighing Model (Pre-processing):**
```bash
python src/train_model.py \\
 --data data/partners.csv \\
 --model_out models/model_reweighed.pkl \\
 --metrics_out reports/metrics_reweighed.json \\
 --fairness_out reports/fairness_reweighed.json \\
 --scores_out data/partners_scores_reweighed.csv \\
 --mitigation reweighing
```

## ⚖️ Comprehensive Fairness Methodology

### Sensitive Attributes
-   **Gender**: female, male, nonbinary
-   **Region**: metro, suburban, rural (proxy for location-based bias)
-   **Role**: driver, merchant (different partner types)

### Fairness Metrics
-   **Performance**: ROC AUC, PR AUC, Brier Score (calibration quality)
-   **Fairness**: Demographic parity, equalized odds, equal opportunity
-   **Detection**: Group-wise positive rates, TPR/FPR differences, score distributions

### Three Mitigation Approaches

| Approach | Method | When Probabilities Change | When Decisions Change | Effectiveness |
|----------|--------|--------------------------|-----------------------|---------------|
| **Baseline** | No mitigation | - | - | None (reference) |
| **Equalized Odds** | Post-processing thresholds | Never | Always | **Excellent** (100x more decisions) |
| **Reweighing** | Training sample weights | Sometimes | Sometimes | Moderate |

### Real Results Summary

**Decision Rates** (out of 12,500 test partners):
-   🎯 **Baseline**: 3 positive decisions (0.0002 rate)
-   ⚖️ **Equalized Odds**: 251 positive decisions (0.0201 rate) 툙2 **100x improvement**
-   ⚖️ **Reweighing**: 5 positive decisions (0.0004 rate) 툙2 2x improvement

**All models maintain ROC AUC = 0.698** (no performance sacrifice)

## ⭐ Nova Score

-   **Formula**: Nova = 300 + (1 - default_probability) × 550
-   **Range**: 300-850 (higher = better creditworthiness, like FICO scores)
-   **Calibration**: Isotonic regression ensures probabilities reflect true default rates
-   **Fairness**: Equalized odds preserves original scores, reweighing produces different scores

**Score Interpretation**:
-   **750-850**: Excellent credit (< 15% default risk)
-   **650-749**: Good credit (15-35% default risk)
-   **500-649**: Fair credit (35-70% default risk)
-   **300-499**: Poor credit (> 70% default risk)

## 📈 Quick Examples

**Partner 27804** (High-risk case):
| Model | Probability | Nova Score | Decision | Notes |
|-------|-------------|------------|----------|-------|
| Baseline | 35.97% | 652.15 | 0 | Below 50% threshold |
| **Equalized Odds** | 35.97% | 652.15 | **1** | **Flagged by fair thresholds** |
| Reweighing | 35.89% | 652.60 | 0 | Slightly lower risk |

**Partner 6144** (Reweighing improvement):
| Model | Probability | Nova Score | Decision | Notes |
|-------|-------------|------------|----------|-------|
| Baseline | 18.26% | 749.57 | 0 | Moderate risk |
| Equalized Odds | 18.26% | 749.57 | 0 | Same assessment |
| **Reweighing** | 17.61% | **753.17** | 0 | **+3.6 Nova points improvement** |

## 🔄 Reproducibility

-   **Deterministic seeds**: All models use seed=42 for consistent results
-   **Fully scripted**: Complete automation from data generation to final reports
-   **Version controlled**: All parameters and configurations documented
-   **Cross-platform**: Tested on Windows, with PowerShell automation included

## 📚 Additional Resources

-   **EXPLANATION.md**: Deep technical dive with model architecture details
-   **PROJECT_NOVA_FLOWCHART.md**: Visual system architecture and workflows
-   **DATA_GENERATION_GUIDE.md**: Advanced data generation scenarios
-   **Jupyter Notebook**: Interactive analysis at `notebooks/01_eda_and_fairness.ipynb`

## 📝 Notes

-   **Synthetic Data**: No real customer data used; fully simulated for hackathon purposes
-   **No PII**: Complete privacy protection through synthetic generation
-   **Extensible**: Easy to add additional Grab-specific features if provided
-   **Production Ready**: Comprehensive error handling, logging, and progress tracking
-   **Research Foundation**: Implements established fairness algorithms (Fairlearn, scikit-learn)

---

**Ready to explore equitable AI?** Run `python run_project.py` and compare three fairness approaches in minutes! 🚀
