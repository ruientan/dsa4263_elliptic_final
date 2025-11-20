"""
Ensemble of XGBoost + GraphSAGE on engineered + weighted configuration.

Assumes you have already run the experiment suite with:
  --configs engineered_weight
for models: xgboost and graphsage

So these files must exist:
  - models/engineered_weight_xgboost_model.pkl
  - models/engineered_weight_graphsage_model.pt
"""

import os
import json
import pickle
import numpy as np
import random

import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

from data_loader import EllipticDataLoader
from models import get_model

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

# Metric helper
def evaluate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred)
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    return metrics


def main():
    CONFIG_NAME = "engineered_weight"
    XGB_PATH = f"models/{CONFIG_NAME}_xgboost_model.pkl"
    SAGE_PATH = f"models/{CONFIG_NAME}_graphsage_model.pt"

    # Check model files exist

    if not os.path.exists(XGB_PATH):
        raise FileNotFoundError(
            f"Could not find {XGB_PATH}. "
            f"Run your experiment suite with config '{CONFIG_NAME}' and XGBoost first."
        )
    if not os.path.exists(SAGE_PATH):
        raise FileNotFoundError(
            f"Could not find {SAGE_PATH}. "
            f"Run your experiment suite with config '{CONFIG_NAME}' and GraphSAGE first."
        )

    print("Using models:")
    print("  XGBoost   :", XGB_PATH)
    print("  GraphSAGE :", SAGE_PATH)


    # Rebuild engineered data and splits

    loader = EllipticDataLoader(data_dir="data")

    # For GraphSAGE: full graph + masks
    data, df_eng = loader.get_pytorch_geometric_data(use_engineered=True)

    # For XGBoost: engineered feature matrix + train/val/test split
    xgb_data = loader.get_feature_matrix_for_xgboost(use_engineered=True)

    X_test = xgb_data["X_test"]
    y_test = xgb_data["y_test"]

    print("\nData shapes:")
    print("  X_test shape:", X_test.shape)
    print("  y_test shape:", y_test.shape)
    print("  GNN test nodes:", int(data.test_mask.sum().item()))

    # Sanity check: test sizes must match
    assert X_test.shape[0] == int(data.test_mask.sum().item()), (
        "Mismatch between XGBoost test size and GraphSAGE test_mask size!"
    )

    # Load XGBoost model + get test probabilities
    with open(XGB_PATH, "rb") as f:
        xgb_model = pickle.load(f)

    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]  # P(y=1)
    xgb_pred = (xgb_prob >= 0.5).astype(int)

    print("\nXGBoost test predictions computed.")

    # Load GraphSAGE model + get test probabilities
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    sage_model = get_model(
        "graphsage",
        num_features=data.num_features,
        hidden_dim=128  # must match your experiment suite
    ).to(device)

    state_dict = torch.load(SAGE_PATH, map_location=device)
    sage_model.load_state_dict(state_dict)
    sage_model.eval()

    with torch.no_grad():
        out = sage_model(data.x.to(device), data.edge_index.to(device))
        # out is log-softmax over 2 classes (as in your GNN training code)
        test_logits = out[data.test_mask.to(device)]
        sage_prob_tensor = torch.exp(test_logits[:, 1])  # P(y=1) = exp(log p1)
        sage_prob = sage_prob_tensor.cpu().numpy()
        sage_pred = (sage_prob >= 0.5).astype(int)

    print("GraphSAGE test predictions computed.")

    # Build ensemble: simple average of probabilities
    # Equal weighting ("poor man's stacking")
    ensemble_prob = (xgb_prob + sage_prob) / 2.0
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)



    # -------------------------------------------------------------------
    # Evaluate all three: XGB, GraphSAGE, Ensemble
    # -------------------------------------------------------------------
    print("\n=== MODEL PERFORMANCE (engineered_weight config) ===")

    xgb_metrics = evaluate_metrics(y_test, xgb_pred, xgb_prob)
    sage_metrics = evaluate_metrics(y_test, sage_pred, sage_prob)
    ens_metrics = evaluate_metrics(y_test, ensemble_pred, ensemble_prob)

    def pretty_print(name, m):
        print(f"\n{name}:")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1        : {m['f1']:.4f}")
        print(f"  AUC-ROC   : {m.get('auc_roc', float('nan')):.4f}")
        print(f"  AUC-PR    : {m.get('auc_pr', float('nan')):.4f}")
        print(f"  TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}")

    pretty_print("XGBoost (engineered_weight)", xgb_metrics)
    pretty_print("GraphSAGE (engineered_weight)", sage_metrics)
    pretty_print("Ensemble = 0.5 * XGB + 0.5 * GraphSAGE", ens_metrics)

    # -------------------------------------------------------------------
    # Save ensemble results
    # -------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    all_results = {
        "xgboost_engineered_weight": xgb_metrics,
        "graphsage_engineered_weight": sage_metrics,
        "ensemble_avg": ens_metrics,
    }
    out_path = "results/ensemble_engineered_weight_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved ensemble metrics to {out_path}")


if __name__ == "__main__":
    main()


