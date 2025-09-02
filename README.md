# Project Nova: Equitable Credit Scoring Engine

An end-to-end, fair, data-driven credit scoring prototype for Grab partners (drivers and merchants). The system simulates realistic operational data, trains ML models, and evaluates/mitigates bias to produce an equitable "Nova Score" (scaled 300–850).

## Contents
- requirements.txt: Python dependencies
- src/generate_data.py: Simulated data generation
- src/train_model.py: Training, evaluation, and fairness analysis/mitigation
- run_project.py: Complete pipeline runner
- notebooks/01_eda_and_fairness.ipynb: EDA and fairness analysis notebook
- .gitignore: Git ignore file

Generated during execution:
- data/: Generated datasets
- models/: Saved models and artifacts
- reports/: Metrics and plots

## Quickstart
1) Create and activate a virtual environment (PowerShell on Windows):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2) Install dependencies:
   pip install -r requirements.txt

3) Run the complete pipeline:
   python run_project.py

## Manual Steps (Alternative)
If you prefer to run steps individually:

1) Generate simulated dataset:
   python src/generate_data.py --n 50000 --seed 42 --out data/partners.csv

2) Train model with fairness optimization:
   python src/train_model.py \
     --data data/partners.csv \
     --model_out models/model.pkl \
     --metrics_out reports/metrics.json \
     --fairness_out reports/fairness.json \
     --scores_out data/partners_with_scores.csv \
     --mitigation equalized_odds

## Fairness Methodology (Summary)
- Sensitive attributes: gender, region (proxy for location), role
- Metrics: AUC, calibration, demographic parity difference, equalized odds difference
- Detection: Report group-wise positive rate, TPR/FPR, and score distributions
- Mitigation options:
  - none: baseline
  - reweighing: sample weights to reduce disparity
  - equalized_odds: post-processing threshold adjustment to enforce EO (via fairlearn)

## Nova Score
- Convert predicted default risk (lower is better) to a score in [300, 850], calibrated via isotonic regression.
- Example: Nova = 300 + (1 - risk) * 550

## Reproducibility
- Deterministic seeds
- Fully scripted generation, training, reporting

## Notes
- This is a simulated dataset for hackathon purposes; no PII.
- Add or adjust features to reflect additional Grab signals if provided.

