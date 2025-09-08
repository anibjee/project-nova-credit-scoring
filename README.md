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
   - 🎯 **Baseline model**: Standard ML (79.8% approval rate)
   - ⚖️ **Fair model**: Aggressive fairness optimization (39.6% approval rate)
   - ⚖️ **Reweighed model**: Balanced approach (78.5% approval rate)
   
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
 --mitigation none \\
 --nova_threshold 700 \\
 --risk_threshold 0.10
```

**Equalized Odds Model (Post-processing):**
```bash
python src/train_model.py \\
 --data data/partners.csv \\
 --model_out models/model_fair.pkl \\
 --metrics_out reports/metrics_fair.json \\
 --fairness_out reports/fairness_fair.json \\
 --scores_out data/partners_scores_fair.csv \\
 --mitigation equalized_odds \\
 --nova_threshold 700 \\
 --risk_threshold 0.10
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

### Three Mitigation Approaches - ACTUAL RESULTS

| Model | Method | Approval Rate | Risk Profile | Business Utility | Fairness Impact |
|-------|--------|---------------|--------------|------------------|----------------|
| **Baseline** | No mitigation | **79.8%** | 5.2% avg risk | **Highest Volume** | Standard |
| **Fair (Equalized Odds)** | Post-processing | **39.6%** | 4.7% avg risk | **Educational Demo** | **Aggressive** |
| **Reweighing** | Pre-processing | **78.5%** | 5.3% avg risk | **Recommended** | **Balanced** |

### Real Results Summary - ACTUAL PERFORMANCE

**Decision Rates** (out of 12,500 test partners):
-   🎯 **Baseline**: 9,972 approvals (**79.8%** rate) - Standard business logic
-   ⚖️ **Fair Model**: 4,949 approvals (**39.6%** rate) - **50% reduction due to aggressive fairness**
-   ⚖️ **Reweighing**: 9,807 approvals (**78.5%** rate) - **Optimal balance** (only 1.3% reduction)

**All models maintain ROC AUC = 0.698** - fairness techniques don't hurt predictive accuracy

**🔧 KEY INSIGHTS FROM THREE-MODEL COMPARISON**: 
- **Baseline (79.8%)**: Maximum volume with standard risk management
- **Fair Model (39.6%)**: Demonstrates over-optimization risk - **educational value**
- **Reweighed (78.5%)**: **Production recommendation** - optimal fairness-utility balance
- **Critical Finding**: Aggressive fairness can exclude 50% of qualified applicants

### 📋 **Decision Output Format**
All models output binary loan decisions as:
- **`1`** = **APPROVE** loan (partner meets credit criteria)
- **`0`** = **REJECT** loan (partner does not meet credit criteria)

Decision logic: `decision = 1 if (nova_score >= 700 AND default_risk <= 10%) else 0`

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

## 📈 Quick Examples - Model Behavior Patterns

**Partner 21282** (High-quality approved case):
| Model | Probability | Nova Score | Decision | Notes |
|-------|-------------|------------|----------|-------|
| Baseline | 4.53% | 825.10 | **1** (APPROVE) | Low risk, excellent credit |
| Fair Model | 4.53% | 825.10 | **1** (APPROVE) | Same probabilities, same decision |
| Reweighing | 4.45% | 825.53 | **1** (APPROVE) | Slightly improved risk assessment |

**Partner 12409** (Fair model conservatism example):
| Model | Probability | Nova Score | Decision | Notes |
|-------|-------------|------------|----------|-------|
| Baseline | 7.58% | 808.28 | **1** (APPROVE) | Good credit, acceptable risk |
| **Fair Model** | 7.58% | 808.28 | **0** (REJECT) | **Conservative thresholds reject qualified applicant** |
| Reweighing | 7.68% | 807.76 | **1** (APPROVE) | Similar to baseline, approved |

**Key Insight**: Fair model's aggressive constraints can exclude qualified applicants, demonstrating why balanced approaches (Reweighing) are preferred for production.

## 🎯 Model Selection Guide

| Business Priority | Recommended Model | Rationale |
|-------------------|-------------------|----------|
| **Maximum Volume** | Baseline (79.8%) | Highest approvals, standard risk management |
| **Production Deployment** | **Reweighed (78.5%)** | **Best balance of fairness and utility** |
| **Fairness Research** | All Three Models | Complete spectrum analysis |
| **Educational Demo** | Fair Model (39.6%) | Shows fairness optimization limits |

**✅ Production Recommendation**: Use the **Reweighed Model** for real-world deployment as it maintains 98.7% of baseline volume while improving fairness across demographic groups.

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
