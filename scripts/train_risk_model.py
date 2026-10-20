"""
SENTINEL — XGBoost Risk Model Trainer
Trains the ML risk ensemble on synthetic training data.

Outputs:
  models/risk_classifier.json    — XGBoost binary classifier
  models/feature_names.json      — ordered feature list
  models/model_metrics.json      — precision, recall, F1, AUC

Run: python scripts/train_risk_model.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_FILE    = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "training_data.jsonl")
MODELS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_FILE   = os.path.join(MODELS_DIR, "risk_classifier.json")
FEATURES_FILE = os.path.join(MODELS_DIR, "feature_names.json")
METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")

os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "tier3_ratio", "tier2_ratio", "seq_length",
    "has_external_url", "has_secret_keyword", "has_shadow_access",
    "has_admin_keyword", "has_curl_external", "has_injection_marker",
    "has_wildcard_perm", "has_public_firewall", "has_unsafe_backup",
    "max_consecutive_tier3", "secret_then_shell",
    "mean_drift_score", "max_drift_score",
    "mean_security_score", "max_security_score",
    "block_rate", "warn_rate",
]


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            feats = rec["features"]
            row = [feats.get(name, 0.0) for name in FEATURE_NAMES]
            X.append(row)
            y.append(rec["label"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train() -> None:
    print("Loading dataset…")
    X, y = load_dataset()
    n_safe = int((y == 0).sum())
    n_adv  = int((y == 1).sum())
    print(f"  Samples: {len(y)}  (safe={n_safe}, adversarial={n_adv})")

    # Class imbalance weight
    scale_pos = n_safe / max(n_adv, 1)

    model = xgb.XGBClassifier(
        n_estimators      = 200,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,
    )

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    print("Running 5-fold cross-validation…")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"  CV F1 scores: {[round(s, 3) for s in cv_scores]}")
    print(f"  CV F1 mean:   {cv_scores.mean():.3f}  (+/- {cv_scores.std():.3f})")

    # ── Final fit on full dataset ─────────────────────────────────────────────
    print("Training final model on full dataset…")
    model.fit(X, y)

    # ── Metrics on training set ───────────────────────────────────────────────
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary")
    auc = roc_auc_score(y, y_prob)

    print("\nTraining-set metrics (expect high — use CV scores for generalisation):")
    print(classification_report(y, y_pred, target_names=["safe", "adversarial"]))
    print(f"  AUC-ROC: {auc:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    model.save_model(MODEL_FILE)
    print(f"\n  Model saved to {MODEL_FILE}")

    with open(FEATURES_FILE, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    metrics = {
        "cv_f1_mean":    round(float(cv_scores.mean()), 4),
        "cv_f1_std":     round(float(cv_scores.std()),  4),
        "cv_f1_scores":  [round(float(s), 4) for s in cv_scores],
        "train_precision": round(float(p),   4),
        "train_recall":    round(float(r),   4),
        "train_f1":        round(float(f1),  4),
        "train_auc":       round(float(auc), 4),
        "n_features":      len(FEATURE_NAMES),
        "n_estimators":    200,
        "n_safe":          n_safe,
        "n_adversarial":   n_adv,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {METRICS_FILE}")

    # ── Feature importance ────────────────────────────────────────────────────
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances),
                    key=lambda x: x[1], reverse=True)
    print("\nTop 10 feature importances:")
    for name, imp in ranked[:10]:
        bar = "#" * int(imp * 40)
        print(f"  {name:<28}  {imp:.4f}  {bar}")


if __name__ == "__main__":
    train()
