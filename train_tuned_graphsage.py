"""
Train tuned GraphSAGE model (engineered_weight) and evaluate on test set.
"""

import os
import json
import time

import numpy as np
import torch
import torch.optim as optim

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from data_loader import EllipticDataLoader
from models import get_model, WeightedBCELoss


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


def train_single_graphsage(data, hidden_dim, lr, weight_decay, epochs, patience, use_class_weights=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = get_model(
        "graphsage",
        num_features=data.num_features,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_class_weights:
        criterion = WeightedBCELoss(data.class_weights.to(device))
    else:
        uniform_weights = torch.ones(2, device=device) / 2
        criterion = WeightedBCELoss(uniform_weights)

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)   # log-probs [N, 2]
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                val_logits = out[data.val_mask]
                val_pred = val_logits.argmax(dim=1)
                val_true = data.y[data.val_mask]
                # assuming out is log-probs as in tune_graphsage.py
                val_prob = torch.exp(val_logits[:, 1])

                val_f1 = f1_score(
                    val_true.cpu().numpy(),
                    val_pred.cpu().numpy(),
                    zero_division=0,
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience // 5:
                print(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time

    # Load best weights
    model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_logits = out[data.test_mask]
        test_pred = test_logits.argmax(dim=1)
        test_true = data.y[data.test_mask]
        test_prob = torch.exp(test_logits[:, 1])

        metrics = evaluate_metrics(
            test_true.cpu().numpy(),
            test_pred.cpu().numpy(),
            test_prob.cpu().numpy(),
        )

    metrics["best_val_f1"] = float(best_val_f1)
    metrics["training_time"] = float(train_time)

    return model, metrics


def main():
    print("=== Training tuned GraphSAGE (engineered_weight) ===")

    # 1. Load data
    loader = EllipticDataLoader(data_dir="data")
    data, df = loader.get_pytorch_geometric_data(use_engineered=True)

    print(f"Num features: {data.num_features}")
    print(f"Train nodes: {int(data.train_mask.sum())}")
    print(f"Val   nodes: {int(data.val_mask.sum())}")
    print(f"Test  nodes: {int(data.test_mask.sum())}")

    # 2. Load best config from tuning
    TUNING_JSON = "results/graphsage_tuning_results.json"
    if not os.path.exists(TUNING_JSON):
        raise FileNotFoundError(
            f"Could not find tuning file: {TUNING_JSON}. "
            "Run tune_graphsage.py first."
        )

    with open(TUNING_JSON, "r") as f:
        tuning_results = json.load(f)

    best_cfg = tuning_results["best_config"]
    print("Using tuned hyperparameters from graphsage_tuning_results.json:")
    for k, v in best_cfg.items():
        print(f"  {k}: {v}")

    # 3. Train GraphSAGE with tuned hyperparams
    model, metrics = train_single_graphsage(
        data,
        hidden_dim=best_cfg["hidden_dim"],
        lr=best_cfg["lr"],
        weight_decay=best_cfg["weight_decay"],
        epochs=best_cfg["epochs"],
        patience=best_cfg["patience"],
        use_class_weights=True,
    )

    print("\n=== TUNED GRAPHSAGE (engineered_weight) TEST PERFORMANCE ===")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"AUC-PR    : {metrics['auc_pr']:.4f}")
    print(
        f"TP={metrics['tp']}, FP={metrics['fp']}, "
        f"FN={metrics['fn']}, TN={metrics['tn']}"
    )

    # 4. Save model + metrics
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model_path = "models/graphsage_engineered_weight_tuned.pt"
    metrics_path = "results/graphsage_engineered_weight_tuned_test_metrics.json"

    torch.save(model.state_dict(), model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved tuned GraphSAGE weights → {model_path}")
    print(f"Saved test metrics → {metrics_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()