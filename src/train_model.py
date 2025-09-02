import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (auc, average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.postprocessing import ThresholdOptimizer


def load_data(path: str):
    df = pd.read_csv(path)
    y = df["defaulted_12m"].astype(int)
    X = df.drop(columns=["defaulted_12m", "partner_id"])
    sensitive = df[["gender", "region", "role"]].copy()
    return X, y, sensitive, df


def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
    return pre


def compute_metrics(y_true, y_prob, y_pred):
    roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    # Precision at operating point from default threshold 0.5 for reference
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    pr_points = list(zip(prec.tolist(), rec.tolist()))
    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "brier": float(bs),
        "pr_curve_points": pr_points[:200],  # limit size
        "default_threshold_pos_rate": float((y_prob >= 0.5).mean()),
    }


def fairness_report(y_true, y_prob, y_pred, sensitive: pd.DataFrame):
    report = {}
    for attr in ["gender", "region", "role"]:
        s = sensitive[attr]
        mf_sr = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=s)
        mf_tpr = MetricFrame(metrics=true_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=s)
        mf_fpr = MetricFrame(metrics=false_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=s)
        # Score distribution summary by group
        groups = {}
        for g in s.unique():
            mask = s == g
            groups[str(g)] = {
                "n": int(mask.sum()),
                "avg_score": float(1 - y_prob[mask].mean()),
                "avg_prob": float(y_prob[mask].mean()),
                "pos_rate": float(mf_sr.by_group[g]),
                "tpr": float(mf_tpr.by_group[g] if g in mf_tpr.by_group else np.nan),
                "fpr": float(mf_fpr.by_group[g] if g in mf_fpr.by_group else np.nan),
            }
        report[attr] = {
            "overall_pos_rate": float(mf_sr.overall),
            "groups": groups,
            "demographic_parity_diff": float(mf_sr.difference(method="between_groups")),
            "tpr_diff": float(mf_tpr.difference(method="between_groups")),
            "fpr_diff": float(mf_fpr.difference(method="between_groups")),
        }
    return report


def to_nova_score(prob_default: np.ndarray) -> np.ndarray:
    # Map probability of default (higher worse) to credit score 300-850 (higher better)
    return 300.0 + (1.0 - prob_default) * 550.0


def train(args):
    X, y, sensitive, raw_df = load_data(args.data)

    X_train, X_test, y_train, y_test, s_train, s_test, raw_train, raw_test = train_test_split(
        X, y, sensitive, raw_df, test_size=0.25, random_state=42, stratify=y
    )

    pre = build_preprocessor(X)

    base_model = HistGradientBoostingClassifier(
        max_depth=None,
        learning_rate=0.06,
        max_iter=350,
        l2_regularization=0.0,
        random_state=42
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", base_model),
    ])

    # Calibrate probabilities for better score reliability
    calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv=3)

    sample_weight = None

    if args.mitigation == "reweighing":
        # Simple reweighing: inverse of group positive rate to balance labels per group
        # Compute weights on train set for a chosen sensitive attribute (region by default)
        attr = "region"
        dfw = s_train.copy()
        dfw["y"] = y_train.values
        weights = np.ones(len(dfw), dtype=float)
        for g in dfw[attr].unique():
            mask = dfw[attr] == g
            pos_rate = dfw.loc[mask, "y"].mean()
            # Avoid division by zero; encourage more weight where positive rate is low
            w = 1.0 / max(pos_rate, 1e-3)
            weights[mask.values] = w
        sample_weight = weights

    calibrated.fit(X_train, y_train, sample_weight=sample_weight)

    y_prob = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_prob, y_pred)

    # Optional post-processing mitigation for equalized odds
    if args.mitigation == "equalized_odds":
        # Use one sensitive attribute for demo; can iterate over multiple
        attr = s_test["region"]
        post = ThresholdOptimizer(
            estimator=calibrated,
            constraints="equalized_odds",
            predict_method="predict_proba",
            prefit=True,
        )
        post.fit(X_train, y_train, sensitive_features=s_train["region"])  # fit uses train
        y_pred = post.predict(X_test, sensitive_features=attr)
        # Get probabilities from the original model since ThresholdOptimizer only adjusts thresholds
        y_prob = calibrated.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob, y_pred)

    fair = fairness_report(y_test, y_prob, y_pred, s_test)

    # Nova Scores
    nova = to_nova_score(y_prob)

    # Persist artifacts
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, args.model_out)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.fairness_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scores_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(args.fairness_out, "w") as f:
        json.dump(fair, f, indent=2)

    out_df = raw_test[["partner_id"]].copy()
    out_df["prob_default"] = y_prob
    out_df["nova_score"] = nova
    out_df.to_csv(args.scores_out, index=False)

    print("Training complete")
    print(f"ROC AUC: {metrics['roc_auc']:.3f} | PR AUC: {metrics['pr_auc']:.3f} | Brier: {metrics['brier']:.4f}")
    print(f"Artifacts written: {args.model_out}, {args.metrics_out}, {args.fairness_out}, {args.scores_out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model_out", type=str, default="models/model.pkl")
    p.add_argument("--metrics_out", type=str, default="reports/metrics.json")
    p.add_argument("--fairness_out", type=str, default="reports/fairness.json")
    p.add_argument("--scores_out", type=str, default="data/partners_with_scores.csv")
    p.add_argument("--mitigation", type=str, choices=["none", "reweighing", "equalized_odds"], default="none")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

