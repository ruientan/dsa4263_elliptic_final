"""
Train tuned XGBoost model (engineered_weight) and evaluate on test set.
"""

import os
import json
import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from data_loader import EllipticDataLoader


def evaluate_metrics(y_true, y_pred, y_prob):
    """Compute evaluation metrics into a dict."""
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    return metrics


def main():
    print("=== Training tuned XGBoost (engineered_weight) ===")


    # 1. Load data (engineered features)
    print("Loading data with engineered features...")
    loader = EllipticDataLoader(data_dir="data")
    xgb_data = loader.get_feature_matrix_for_xgboost(use_engineered=True)

    X_train, y_train = xgb_data["X_train"], xgb_data["y_train"]
    X_val, y_val = xgb_data["X_val"], xgb_data["y_val"]
    X_test, y_test = xgb_data["X_test"], xgb_data["y_test"]

    # Combine train + val for final training
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    print("Train+Val shape:", X_train_full.shape)
    print("Test shape     :", X_test.shape)


    # 2. Define tuned hyperparameters (from JSON file)

    TUNING_JSON = "results/xgb_tuning_results.json"

    if not os.path.exists(TUNING_JSON):
        raise FileNotFoundError(
            f"Could not find tuning file: {TUNING_JSON}. "
            "Run tune_xgboost.py first."
        )

    with open(TUNING_JSON, "r") as f:
        tuning_results = json.load(f)

    best_params = tuning_results["best_params"]

    print("Using tuned hyperparameters from tuning_results.json:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")


    # 3. Create and train XGBoost model
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        **best_params,
    )

    print("\nFitting tuned model on Train+Val...")
    model.fit(X_train_full, y_train_full)
    print("Training complete.")

    # 4. Evaluate on test set
    print("\nEvaluating on held-out test set...")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = evaluate_metrics(y_test, y_pred, y_prob)

    print("\n=== TUNED XGBOOST (engineered_weight) TEST PERFORMANCE ===")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"AUC-PR    : {metrics['auc_pr']:.4f}")
    print(
        f"TP={metrics['tp']}, FP={metrics['fp']}, "
        f"FN={metrics['fn']}, TN={metrics['tn']}"
    )

    # 5. Save model + metrics 
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model_path = "models/xgboost_engineered_weight_tuned.pkl"
    metrics_path = "results/xgb_engineered_weight_tuned_test_metrics.json"

    with open(model_path, "wb") as f:
        import pickle

        pickle.dump(model, f)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved tuned model → {model_path}")
    print(f"Saved test metrics → {metrics_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()