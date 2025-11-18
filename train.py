import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import pickle
import json

from data_loader import EllipticDataLoader
from feature_eng import FeatureEngineer
from models import get_model, FocalLoss, WeightedBCELoss


def evaluate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        metrics['auc_pr'] = average_precision_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def train_gnn(model_name, data, epochs=200, lr=0.01, patience=30, use_class_weights=True):
    print(f"Training {model_name}...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, num_features=data.num_features)
    model = model.to(device)
    data = data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    if use_class_weights:
        criterion = WeightedBCELoss(data.class_weights.to(device))
    else:
        uniform_weights = torch.ones(2) / 2
        criterion = WeightedBCELoss(uniform_weights.to(device))

    model.train()
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

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

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

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

    torch.save(best_model_state, f'models/{model_name}_model.pt')
    print(f"{model_name}: F1={test_metrics['f1']:.3f}, Precision={test_metrics['precision']:.3f}, Recall={test_metrics['recall']:.3f}")

    return test_metrics


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, use_class_weights=True):
    print("Training XGBoost...")

    scale_pos_weight = 1
    if use_class_weights:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'scale_pos_weight': scale_pos_weight,
        'early_stopping_rounds': 20
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_metrics(y_test, y_pred, y_prob)

    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print(f"XGBoost: F1={test_metrics['f1']:.3f}, Precision={test_metrics['precision']:.3f}, Recall={test_metrics['recall']:.3f}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['gcn', 'gat', 'graphsage', 'xgboost'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--use_engineered_features', action='store_true')
    parser.add_argument('--no_class_weights', action='store_true')
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)

    loader = EllipticDataLoader(data_dir='data')
    use_class_weights = not args.no_class_weights

    if args.use_engineered_features:
        print("Using engineered features")
        data, df = loader.get_pytorch_geometric_data(use_engineered=True)
    else:
        print("Using baseline features")
        data, df = loader.get_pytorch_geometric_data(use_engineered=False)

    results = {}

    for model_name in args.models:
        if model_name == 'xgboost':
            # Use masks from data object (move to CPU first)
            X_train = data.x[data.train_mask].cpu().numpy()
            y_train = data.y[data.train_mask].cpu().numpy()

            X_val = data.x[data.val_mask].cpu().numpy()
            y_val = data.y[data.val_mask].cpu().numpy()

            X_test = data.x[data.test_mask].cpu().numpy()
            y_test = data.y[data.test_mask].cpu().numpy()

            results['xgboost'] = train_xgboost(
                X_train, y_train, X_val, y_val, X_test, y_test,
                use_class_weights=use_class_weights
            )
        else:
            results[model_name] = train_gnn(
                model_name, data, epochs=args.epochs,
                use_class_weights=use_class_weights
            )

    with open('models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nAll models trained successfully")


if __name__ == '__main__':
    main()