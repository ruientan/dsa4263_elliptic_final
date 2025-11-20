#!/usr/bin/env python3
"""
Comprehensive fraud detection experiment suite
Runs all 4 configurations: {baseline, engineered} × {no weights, with weights}
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import pickle
import json
import random
from datetime import datetime

from data_loader import EllipticDataLoader
from feature_eng import FeatureEngineer
from models import get_model, FocalLoss, WeightedBCELoss

SEED = 42

# Python RNG
random.seed(SEED)

# NumPy RNG
np.random.seed(SEED)

# PyTorch RNG
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Make cuDNN deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Python hash seed (important!)
os.environ["PYTHONHASHSEED"] = str(SEED)

def evaluate_metrics(y_true, y_pred, y_prob=None):
    """Compute comprehensive evaluation metrics"""
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


def train_gnn(model_name, data, epochs=200, lr=0.01, patience=40, use_class_weights=True, verbose=True, config_name=''):
    """Train a GNN model with early stopping"""
    if verbose:
        print(f"\nTraining {model_name}...")
        print(f"  Features: {data.num_features}")
        print(f"  Training samples: {data.train_mask.sum().item()}")
        print(f"  Class weights: {use_class_weights}")

    # Use GPU if available (RTX 3090 has plenty of VRAM)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, num_features=data.num_features, hidden_dim=128)
    model = model.to(device)
    data = data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    if use_class_weights:
        criterion = WeightedBCELoss(data.class_weights.to(device))
    else:
        uniform_weights = torch.ones(2) / 2
        criterion = WeightedBCELoss(uniform_weights.to(device))

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

        # Validation
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

                if patience_counter >= patience // 5:  # Check every 5 epochs
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break

    training_time = time.time() - training_start

    # Load best model and evaluate on test set
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

    # Save model with config name
    os.makedirs('models', exist_ok=True)
    model_filename = f'models/{config_name}_{model_name}_model.pt' if config_name else f'models/{model_name}_model.pt'
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


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test,
                  use_class_weights=True, verbose=True, config_name=''):
    """Train XGBoost classifier"""
    if verbose:
        print("\nTraining XGBoost...")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Training samples: {len(y_train)}")
        print(f"  Class weights: {use_class_weights}")

    scale_pos_weight = 1
    if use_class_weights:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        if verbose:
            print(f"  Scale pos weight: {scale_pos_weight:.2f}")

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'scale_pos_weight': scale_pos_weight,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'early_stopping_rounds': 30,
        'random_state': SEED,
        'seed': SEED,
    }

    training_start = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    training_time = time.time() - training_start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_metrics(y_test, y_pred, y_prob)
    test_metrics['training_time'] = training_time

    # Save model with config name
    os.makedirs('models', exist_ok=True)
    model_filename = f'models/{config_name}_xgboost_model.pkl' if config_name else 'models/xgboost_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

    if verbose:
        print(f"  XGBoost: F1={test_metrics['f1']:.4f}, "
              f"Precision={test_metrics['precision']:.4f}, "
              f"Recall={test_metrics['recall']:.4f}, "
              f"AUC-ROC={test_metrics.get('auc_roc', 0):.4f}")

    return test_metrics


def run_single_configuration(config_name, use_engineered, use_class_weights,
                             models_to_run, epochs=200):
    """Run experiments for a single configuration"""
    print("\n" + "=" * 80)
    print(f"CONFIGURATION: {config_name}")
    print(f"  Engineered features: {use_engineered}")
    print(f"  Class weights: {use_class_weights}")
    print("=" * 80)

    loader = EllipticDataLoader(data_dir='data')

    results = {
        'config_name': config_name,
        'use_engineered_features': use_engineered,
        'use_class_weights': use_class_weights,
        'models': {}
    }

    # Load data for GNN models
    if any(m in models_to_run for m in ['gcn', 'gat', 'graphsage']):
        data, df = loader.get_pytorch_geometric_data(use_engineered=use_engineered)

        for model_name in ['gcn', 'gat', 'graphsage']:
            if model_name in models_to_run:
                try:
                    metrics = train_gnn(
                        model_name, data,
                        epochs=epochs,
                        use_class_weights=use_class_weights,
                        config_name=config_name
                    )
                    results['models'][model_name] = metrics
                except Exception as e:
                    print(f"  ERROR training {model_name}: {e}")
                    results['models'][model_name] = {'error': str(e)}

    # Train XGBoost
    if 'xgboost' in models_to_run:
        try:
            xgb_data = loader.get_feature_matrix_for_xgboost(use_engineered=use_engineered)
            metrics = train_xgboost(
                xgb_data['X_train'], xgb_data['y_train'],
                xgb_data['X_val'], xgb_data['y_val'],
                xgb_data['X_test'], xgb_data['y_test'],
                use_class_weights=use_class_weights,
                config_name=config_name
            )
            results['models']['xgboost'] = metrics
        except Exception as e:
            print(f"  ERROR training XGBoost: {e}")
            results['models']['xgboost'] = {'error': str(e)}

    return results


# def create_comparison_table(all_results):
#     """Create a comparison table from all results"""
#     rows = []

#     for config_result in all_results:
#         config_name = config_result['config_name']

#         for model_name, metrics in config_result['models'].items():
#             if 'error' in metrics:
#                 continue

#             row = {
#                 'Configuration': config_name,
#                 'Model': model_name,
#                 'F1': metrics['f1'],
#                 'Precision': metrics['precision'],
#                 'Recall': metrics['recall'],
#                 'AUC-ROC': metrics.get('auc_roc', np.nan),
#                 'AUC-PR': metrics.get('auc_pr', np.nan),
#                 'Training_Time': metrics.get('training_time', np.nan),
#                 'TP': metrics.get('tp', 0),
#                 'TN': metrics.get('tn', 0),
#                 'FP': metrics.get('fp', 0),
#                 'FN': metrics.get('fn', 0)
#             }
#             rows.append(row)

#     df = pd.DataFrame(rows)

#     # Only sort if we have results
#     if len(df) > 0 and 'F1' in df.columns:
#         df = df.sort_values(['F1', 'AUC-ROC'], ascending=False)

#     return df


def create_comparison_table(all_results):
    """Create a comparison table from all results"""
    rows = []

    for config_result in all_results:
        config_name = config_result['config_name']

        for model_name, metrics in config_result['models'].items():
            if 'error' in metrics:
                continue

            row = {
                'Configuration': config_name,
                'Model': model_name,
                'F1': metrics['f1'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'AUC-ROC': metrics.get('auc_roc', np.nan),
                'AUC-PR': metrics.get('auc_pr', np.nan),
                'Training_Time': metrics.get('training_time', np.nan),
                'TP': metrics.get('tp', 0),
                'TN': metrics.get('tn', 0),
                'FP': metrics.get('fp', 0),
                'FN': metrics.get('fn', 0)
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Only sort if we have results
    if len(df) > 0 and 'Recall' in df.columns:
        # Primary: Recall (catch illicit wallets)
        # Secondary: AUC-PR (handles imbalance)
        # Tertiary: F1 (balance between precision & recall)
        df = df.sort_values(['Recall', 'AUC-PR', 'F1'], ascending=False)

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive fraud detection experiments'
    )
    parser.add_argument('--models', nargs='+',
                       default=['gcn', 'gat', 'graphsage', 'xgboost'],
                       choices=['gcn', 'gat', 'graphsage', 'xgboost'],
                       help='Models to train')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum training epochs')
    parser.add_argument('--configs', nargs='+',
                       choices=['baseline_noweight', 'baseline_weight',
                               'engineered_noweight', 'engineered_weight', 'all'],
                       default=['all'],
                       help='Configurations to run')

    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE FRAUD DETECTION EXPERIMENT SUITE")
    print("=" * 80)
    print(f"Models: {args.models}")
    print(f"Max epochs: {args.epochs}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")
    print("=" * 80)

    # Define all configurations
    configurations = [
        ('baseline_noweight', False, False),
        ('baseline_weight', False, True),
        ('engineered_noweight', True, False),
        ('engineered_weight', True, True)
    ]

    # Filter configurations if specified
    if 'all' not in args.configs:
        configurations = [c for c in configurations if c[0] in args.configs]

    all_results = []

    # Run each configuration
    for config_name, use_engineered, use_class_weights in configurations:
        try:
            result = run_single_configuration(
                config_name, use_engineered, use_class_weights,
                args.models, args.epochs
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR in configuration {config_name}: {e}")
            import traceback
            traceback.print_exc()

    # Create comparison table
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)

    df = create_comparison_table(all_results)
    print(df.to_string(index=False))

    # Save results
    os.makedirs('results', exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed JSON
    with open(f'results/full_experiments_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save comparison table
    df.to_csv(f'results/comparison_{timestamp}.csv', index=False)

    with open(f'results/summary_{timestamp}.txt', 'w') as f:
        f.write("FRAUD DETECTION EXPERIMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        if len(df) > 0:
            best = df.iloc[0]
            f.write(f"Best Configuration (by Recall → AUC-PR → F1): {best['Configuration']}\n")
            f.write(f"Best Model: {best['Model']}\n")
            f.write(f"Recall: {best['Recall']:.4f}\n")
            f.write(f"AUC-PR: {best['AUC-PR']:.4f}\n")
            f.write(f"F1: {best['F1']:.4f}\n")
            f.write(f"AUC-ROC: {best['AUC-ROC']:.4f}\n")
        else:
            f.write("No successful model runs\n")

    print(f"\nResults saved to results/ directory")
    if len(df) > 0:
        best = df.iloc[0]
        print("\nBest performing configuration (by Recall → AUC-PR → F1):")
        print(f"  {best['Configuration']} - {best['Model']}")
        print(f"  Recall: {best['Recall']:.4f}, AUC-PR: {best['AUC-PR']:.4f}")
        print(f"  F1: {best['F1']:.4f}, AUC-ROC: {best['AUC-ROC']:.4f}")
    else:
        print("\nNo successful model runs to report")

    # # Save summary
    # with open(f'results/summary_{timestamp}.txt', 'w') as f:
    #     f.write("FRAUD DETECTION EXPERIMENT RESULTS\n")
    #     f.write("=" * 80 + "\n\n")
    #     f.write(df.to_string(index=False))
    #     f.write("\n\n")
    #     if len(df) > 0:
    #         f.write(f"Best Configuration: {df.iloc[0]['Configuration']}\n")
    #         f.write(f"Best Model: {df.iloc[0]['Model']}\n")
    #         f.write(f"Best F1 Score: {df.iloc[0]['F1']:.4f}\n")
    #         f.write(f"Best AUC-ROC: {df.iloc[0]['AUC-ROC']:.4f}\n")
    #     else:
    #         f.write("No successful model runs\n")

    # print(f"\nResults saved to results/ directory")
    # if len(df) > 0:
    #     print(f"\nBest performing configuration:")
    #     print(f"  {df.iloc[0]['Configuration']} - {df.iloc[0]['Model']}")
    #     print(f"  F1: {df.iloc[0]['F1']:.4f}, AUC-ROC: {df.iloc[0]['AUC-ROC']:.4f}")
    # else:
    #     print("\nNo successful model runs to report")


if __name__ == '__main__':
    main()
