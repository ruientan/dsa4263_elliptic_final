"""
Ensemble BASELINE GraphSAGE + BASELINE XGBoost for the 'engineered_weight' setting.

- Loads saved baseline models:
    - GraphSAGE: models/engineered_weight_graphsage_model.pt
    - XGBoost:   models/engineered_weight_xgboost_model.pkl
- Uses EllipticDataLoader with use_engineered=True
- Averages probabilities from both models on the TEST set
- Computes metrics and saves them to:
    results/ensemble_engineered_weight_baseline_graphsage_xgb_test_metrics.json
"""

import os
import json
import pickle
import numpy as np
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from data_loader import EllipticDataLoader
from models import get_model


# Helper: compute metrics
def evaluate_metrics(y_true, y_pred, y_prob):
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
    print("=== Ensemble BASELINE GraphSAGE + BASELINE XGBoost (engineered_weight) ===")

    # Load data (engineered features + class weights)
    loader = EllipticDataLoader(data_dir="data")

    # PyG data for GraphSAGE (engineered features)
    data, _ = loader.get_pytorch_geometric_data(use_engineered=True)

    # Tabular data for XGBoost (engineered features)
    xgb_data = loader.get_feature_matrix_for_xgboost(use_engineered=True)
    X_test, y_test_xgb = xgb_data["X_test"], xgb_data["y_test"]

    # Load baseline XGBoost model
    xgb_model_path = "models/engineered_weight_xgboost_model.pkl"
    if not os.path.exists(xgb_model_path):
        raise FileNotFoundError(f"Missing XGBoost baseline model: {xgb_model_path}")

    with open(xgb_model_path, "rb") as f:
        xgb_model = pickle.load(f)

    print(f"Loaded baseline XGBoost model from {xgb_model_path}")

    # XGBoost test probabilities for positive class (illicit)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # Load baseline GraphSAGE model
    graphsage_model_path = "models/engineered_weight_graphsage_model.pt"
    if not os.path.exists(graphsage_model_path):
        raise FileNotFoundError(f"Missing GraphSAGE baseline model: {graphsage_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    hidden_dim = 128

    graphsage_model = get_model(
        "graphsage",
        num_features=data.num_features,
        hidden_dim=hidden_dim,
    ).to(device)

    state_dict = torch.load(graphsage_model_path, map_location=device)
    graphsage_model.load_state_dict(state_dict)
    graphsage_model.eval()

    print(f"Loaded baseline GraphSAGE model from {graphsage_model_path}")

    # GraphSAGE test probabilities
    with torch.no_grad():
        out = graphsage_model(data.x, data.edge_index)  # assumed log-probs [N, 2]
        test_logits = out[data.test_mask]
        test_pred_gs = test_logits.argmax(dim=1)
        test_true_gs = data.y[data.test_mask]
        # assuming outputs are log-probabilities; exp to get probabilities
        y_prob_gs = torch.exp(test_logits[:, 1]).cpu().numpy()
        y_test_gs = test_true_gs.cpu().numpy()

    # Sanity check: label alignment
    if not np.array_equal(y_test_gs, y_test_xgb):
        raise ValueError(
            "Test labels from GraphSAGE and XGBoost do not match. "
            "Check that your EllipticDataLoader uses consistent splits/order "
            "for both PyG data and XGBoost feature matrices."
        )

    y_test = y_test_xgb  # or y_test_gs, they should be identical here

    # Ensemble: simple average of probabilities
    w_gs = 0.5  # weight for GraphSAGE
    w_xgb = 0.5  # weight for XGBoost

    y_prob_ens = w_gs * y_prob_gs + w_xgb * y_prob_xgb
    y_pred_ens = (y_prob_ens >= 0.5).astype(int)

    ensemble_metrics = evaluate_metrics(y_test, y_pred_ens, y_prob_ens)

    print("\n=== ENSEMBLE (Baseline GraphSAGE + Baseline XGBoost) TEST PERFORMANCE ===")
    print(f"Precision : {ensemble_metrics['precision']:.4f}")
    print(f"Recall    : {ensemble_metrics['recall']:.4f}")
    print(f"F1        : {ensemble_metrics['f1']:.4f}")
    print(f"AUC-ROC   : {ensemble_metrics['auc_roc']:.4f}")
    print(f"AUC-PR    : {ensemble_metrics['auc_pr']:.4f}")
    print(
        f"TP={ensemble_metrics['tp']}, FP={ensemble_metrics['fp']}, "
        f"FN={ensemble_metrics['fn']}, TN={ensemble_metrics['tn']}"
    )

    # Save ensemble metrics
    os.makedirs("results", exist_ok=True)
    out_json = "results/ensemble_graphsage_xgb_test_metrics.json"
    with open(out_json, "w") as f:
        json.dump(ensemble_metrics, f, indent=2)

    print(f"\nSaved ensemble metrics → {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()