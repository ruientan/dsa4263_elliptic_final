"""
Hyperparameter tuning for GraphSAGE on the engineered_weight setting.

- Uses EllipticDataLoader with use_engineered=True (engineered features)
- Uses class weights (same idea as engineered_weight config)
- Tunes a small grid of hyperparameters for GraphSAGE
- Selects best config by validation F1, then reports test metrics
"""

import os
import json
import time
import itertools

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


# ---------- Helper: metric computation (same style as your suite) ----------
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
    metrics["confusion_matrix"] = cm.tolist()
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    return metrics


# ---------- Core: train a single GraphSAGE with given hyperparams ----------
def train_graphsage_with_params(
    data,
    hidden_dim=128,
    lr=0.01,
    weight_decay=5e-4,
    epochs=200,
    patience=40,
    use_class_weights=True,
    verbose=False,
):
    """
    Train GraphSAGE with a specific set of hyperparameters.
    Returns:
      - best_val_f1
      - test_metrics (dict)
      - epochs_trained
      - training_time
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = get_model(
        "graphsage",
        num_features=data.num_features,
        hidden_dim=hidden_dim,
    ).to(device)

    data = data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_class_weights:
        criterion = WeightedBCELoss(data.class_weights.to(device))
    else:
        uniform_weights = torch.ones(2) / 2
        criterion = WeightedBCELoss(uniform_weights.to(device))

    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    training_start = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validate every 5 epochs (same pattern as your suite)
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                val_logits = out[data.val_mask]
                val_pred = val_logits.argmax(dim=1)
                val_true = data.y[data.val_mask]
                val_prob = torch.exp(val_logits[:, 1])

                val_metrics = evaluate_metrics(
                    val_true.cpu().numpy(),
                    val_pred.cpu().numpy(),
                    val_prob.cpu().numpy(),
                )
                val_f1 = val_metrics["f1"]

            if verbose and epoch % 20 == 0:
                print(
                    f"  Epoch {epoch:3d} | "
                    f"Loss={loss.item():.4f} | Val F1={val_f1:.4f}"
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience // 5:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

    training_time = time.time() - training_start

    # Evaluate best model on TEST set
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_logits = out[data.test_mask]
        test_pred = test_logits.argmax(dim=1)
        test_true = data.y[data.test_mask]
        test_prob = torch.exp(test_logits[:, 1])

        test_metrics = evaluate_metrics(
            test_true.cpu().numpy(),
            test_pred.cpu().numpy(),
            test_prob.cpu().numpy(),
        )

    test_metrics["best_val_f1"] = best_val_f1
    test_metrics["training_time"] = training_time

    return best_val_f1, test_metrics, epoch + 1


# ---------- Main tuning loop ----------
def main():
    print("=" * 80)
    print("GraphSAGE Hyperparameter Tuning (engineered_weight)")
    print("=" * 80)

    # 1. Load data once (engineered features + class weights)
    loader = EllipticDataLoader(data_dir="data")
    # use_engineered=True, and we'll conceptually be in the "engineered_weight" setting
    data, df = loader.get_pytorch_geometric_data(use_engineered=True)

    print(f"Num features: {data.num_features}")
    print(f"Train nodes: {int(data.train_mask.sum())}")
    print(f"Val   nodes: {int(data.val_mask.sum())}")
    print(f"Test  nodes: {int(data.test_mask.sum())}")

    # 2. Define a SMALL hyperparameter grid (so it finishes in reasonable time)
    hidden_dims = [64, 128]
    lrs = [0.005, 0.01]
    weight_decays = [5e-5, 5e-4]
    epochs = 200
    patience = 40

    search_space = list(itertools.product(hidden_dims, lrs, weight_decays))
    print(f"\nTotal configs to try: {len(search_space)}")
    print("(hidden_dim, lr, weight_decay) combinations:")

    for cfg in search_space:
        print(" ", cfg)

    all_results = []
    best_config = None
    best_val_f1 = -1.0
    best_test_metrics = None

    # 3. Loop over hyperparameter combinations
    for i, (hidden_dim, lr, wd) in enumerate(search_space, start=1):
        print("\n" + "-" * 80)
        print(f"Config {i}/{len(search_space)}")
        print(f"  hidden_dim   = {hidden_dim}")
        print(f"  lr           = {lr}")
        print(f"  weight_decay = {wd}")

        val_f1, test_metrics, epochs_trained = train_graphsage_with_params(
            data,
            hidden_dim=hidden_dim,
            lr=lr,
            weight_decay=wd,
            epochs=epochs,
            patience=patience,
            use_class_weights=True,  # 'engineered_weight' setup
            verbose=False,
        )

        print(f"  -> Best Val F1: {val_f1:.4f}")
        print(
            f"  -> Test F1: {test_metrics['f1']:.4f}, "
            f"Test Recall: {test_metrics['recall']:.4f}, "
            f"Test AUC-PR: {test_metrics.get('auc_pr', float('nan')):.4f}"
        )

        # Save this run's info
        result_row = {
            "hidden_dim": hidden_dim,
            "lr": lr,
            "weight_decay": wd,
            "best_val_f1": val_f1,
            "epochs_trained": epochs_trained,
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_auc_pr": test_metrics.get("auc_pr", np.nan),
            "test_auc_roc": test_metrics.get("auc_roc", np.nan),
            "tp": test_metrics.get("tp", 0),
            "fp": test_metrics.get("fp", 0),
            "fn": test_metrics.get("fn", 0),
            "tn": test_metrics.get("tn", 0),
            "training_time": test_metrics.get("training_time", np.nan),
        }
        all_results.append(result_row)

        # Track best by validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_config = {
                "hidden_dim": hidden_dim,
                "lr": lr,
                "weight_decay": wd,
                "epochs": epochs,
                "patience": patience,
            }
            best_test_metrics = test_metrics

    # 4. Save results
    os.makedirs("results", exist_ok=True)

    # Save all configs + performance
    import pandas as pd

    df = pd.DataFrame(all_results)
    df = df.sort_values(["best_val_f1", "test_auc_pr", "test_f1"], ascending=False)
    csv_path = "results/graphsage_tuning_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved full tuning table to {csv_path}")

    # Save best config + metrics
    best_summary = {
        "best_config": best_config,
        "best_val_f1": best_val_f1,
        "best_test_metrics": best_test_metrics,
    }
    json_path = "results/graphsage_tuning_results.json"
    with open(json_path, "w") as f:
        json.dump(best_summary, f, indent=2)
    print(f"Saved best config summary to {json_path}")

    # 5. Print a nice summary
    print("\n" + "=" * 80)
    print("Best GraphSAGE config (by validation F1)")
    print("=" * 80)
    print(best_config)
    print("\nTest performance of best config:")
    print(f"  Precision : {best_test_metrics['precision']:.4f}")
    print(f"  Recall    : {best_test_metrics['recall']:.4f}")
    print(f"  F1        : {best_test_metrics['f1']:.4f}")
    print(f"  AUC-PR    : {best_test_metrics.get('auc_pr', float('nan')):.4f}")
    print(f"  AUC-ROC   : {best_test_metrics.get('auc_roc', float('nan')):.4f}")
    print(
        f"  TP={best_test_metrics.get('tp', 0)}, "
        f"FP={best_test_metrics.get('fp', 0)}, "
        f"FN={best_test_metrics.get('fn', 0)}, "
        f"TN={best_test_metrics.get('tn', 0)}"
    )


if __name__ == "__main__":
    main()
