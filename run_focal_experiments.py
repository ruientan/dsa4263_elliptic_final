#!/usr/bin/env python3
"""
Focal loss vs cross entropy comparison
Tests focal loss as alternative to weighted BCE for class imbalance
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix

from data_loader import EllipticDataLoader
from models import get_model, FocalLoss


def evaluate_metrics(y_true, y_pred, y_prob=None):
    """Compute evaluation metrics"""
    metrics = {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
        metrics['auc_pr'] = float(average_precision_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])

    return metrics


def train_gnn_focal(model_name, data, epochs=200, lr=0.01, patience=40,
                    use_focal_loss=True, alpha=0.25, gamma=2.0,
                    verbose=True, config_name=''):
    """Train GNN model with focal loss option"""
    if verbose:
        print(f"\nTraining {model_name}...")
        print(f"  Features: {data.num_features}")
        print(f"  Training samples: {data.train_mask.sum().item()}")
        print(f"  Loss: {'Focal Loss' if use_focal_loss else 'Cross Entropy'}")
        if use_focal_loss:
            print(f"  Alpha: {alpha}, Gamma: {gamma}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, num_features=data.num_features, hidden_dim=128)
    model = model.to(device)
    data = data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    if use_focal_loss:
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    training_start = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                val_pred = out[data.val_mask].argmax(dim=1)
                val_true = data.y[data.val_mask]
                val_prob = torch.exp(out[data.val_mask][:, 1])

                val_metrics = evaluate_metrics(
                    val_true.cpu().numpy(),
                    val_pred.cpu().numpy(),
                    val_prob.cpu().numpy()
                )

                val_f1 = val_metrics['f1']

                if verbose and epoch % 20 == 0:
                    print(f"  Epoch {epoch:3d}: Loss={loss.item():.4f}, Val F1={val_f1:.4f}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience // 5:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break

    training_time = time.time() - training_start

    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_pred = out[data.test_mask].argmax(dim=1)
        test_true = data.y[data.test_mask]
        test_prob = torch.exp(out[data.test_mask][:, 1])

        test_metrics = evaluate_metrics(
            test_true.cpu().numpy(),
            test_pred.cpu().numpy(),
            test_prob.cpu().numpy()
        )

    # Save model
    os.makedirs('models', exist_ok=True)
    model_filename = f'models/{config_name}_{model_name}_model.pt' if config_name else f'models/{model_name}_focal_model.pt'
    torch.save(best_model_state, model_filename)

    if verbose:
        print(f"  {model_name}: F1={test_metrics['f1']:.4f}, "
              f"Precision={test_metrics['precision']:.4f}, "
              f"Recall={test_metrics['recall']:.4f}, "
              f"AUC-ROC={test_metrics.get('auc_roc', 0):.4f}")

    test_metrics['training_time'] = training_time
    test_metrics['best_val_f1'] = best_val_f1
    test_metrics['epochs_trained'] = epoch + 1

    return test_metrics


def main():
    print("=" * 80)
    print("FOCAL LOSS vs CROSS ENTROPY COMPARISON")
    print("=" * 80)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}\n")

    loader = EllipticDataLoader(data_dir='data')

    # Compute alpha from class weights
    data_temp, _ = loader.get_pytorch_geometric_data(use_engineered=False)
    class_weight_pos = data_temp.class_weights[1].item()
    class_weight_neg = data_temp.class_weights[0].item()
    alpha_weight = class_weight_pos / (class_weight_pos + class_weight_neg)
    print(f"Alpha (normalized class weight): {alpha_weight:.4f}\n")

    # Experiment matrix - focal loss only (CE baselines in run_full_experiments.py)
    experiments = [
        # GraphSAGE
        ('graphsage', False, True, alpha_weight, 'graphsage_baseline_focal'),
        ('graphsage', True, True, alpha_weight, 'graphsage_engineered_focal'),

        # GAT
        ('gat', False, True, alpha_weight, 'gat_baseline_focal'),
        ('gat', True, True, alpha_weight, 'gat_engineered_focal'),

        # GCN
        ('gcn', False, True, alpha_weight, 'gcn_baseline_focal'),
        ('gcn', True, True, alpha_weight, 'gcn_engineered_focal'),
    ]

    all_results = []

    for model_name, use_engineered, use_focal, alpha, exp_name in experiments:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {exp_name}")
        print(f"  Model: {model_name.upper()}")
        print(f"  Features: {'engineered' if use_engineered else 'baseline'}")
        print(f"  Loss: {'Focal Loss' if use_focal else 'Cross Entropy'}")
        print("=" * 80)

        try:
            # Load data
            data, df = loader.get_pytorch_geometric_data(use_engineered=use_engineered)

            # Train model
            metrics = train_gnn_focal(
                model_name=model_name,
                data=data,
                epochs=200,
                lr=0.01,
                patience=40,
                use_focal_loss=use_focal,
                alpha=alpha,
                gamma=2.0,
                verbose=True,
                config_name=exp_name
            )

            # Store results
            result = {
                'experiment': exp_name,
                'model': model_name,
                'features': 'engineered' if use_engineered else 'baseline',
                'loss_type': 'Focal' if use_focal else 'CE',
                'alpha': alpha if use_focal else None,
                **metrics
            }
            all_results.append(result)

            print(f"\n  Results: F1={metrics['f1']:.4f}, "
                  f"Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}, "
                  f"AUC-ROC={metrics.get('auc_roc', 0):.4f}")

        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results dataframe
    df_results = pd.DataFrame(all_results)

    if len(df_results) > 0:
        df_results = df_results.sort_values('f1', ascending=False)

        # Save CSV
        csv_path = f'results/focal_experiments_{timestamp}.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"Saved detailed results to {csv_path}")

        # Print results by model
        print("\n" + "=" * 80)
        print("FOCAL LOSS RESULTS")
        print("=" * 80)

        for model in df_results['model'].unique():
            model_results = df_results[df_results['model'] == model]
            print(f"\n{model.upper()}:")
            print("-" * 80)

            result_cols = ['experiment', 'features', 'f1', 'precision', 'recall', 'auc_roc']
            print(model_results[result_cols].to_string(index=False))

        # Top performers
        print("\n" + "=" * 80)
        print("TOP RESULTS")
        print("=" * 80)
        top_cols = ['experiment', 'model', 'features', 'f1', 'precision', 'recall', 'auc_roc']
        print(df_results.head()[top_cols].to_string(index=False))

        # Save summary
        with open(f'results/focal_summary_{timestamp}.txt', 'w') as f:
            f.write("FOCAL LOSS EXPERIMENT RESULTS\n")
            f.write("=" * 80 + "\n\n")

            for model in df_results['model'].unique():
                model_results = df_results[df_results['model'] == model]
                f.write(f"\n{model.upper()}:\n")
                f.write("-" * 80 + "\n")
                f.write(model_results[result_cols].to_string(index=False))
                f.write("\n\n")

            f.write("\nTOP RESULTS:\n")
            f.write("=" * 80 + "\n")
            f.write(df_results.head()[top_cols].to_string(index=False))
            f.write("\n")

        print(f"\nSaved summary to results/focal_summary_{timestamp}.txt")

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
