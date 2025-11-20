import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import average_precision_score, make_scorer
from data_loader import EllipticDataLoader
import random
import os

SEED = 42

# Python RNG
random.seed(SEED)

# NumPy RNG
np.random.seed(SEED)

# Python hashing determinism
os.environ["PYTHONHASHSEED"] = str(SEED)

print("Loading data...")
loader = EllipticDataLoader(data_dir="data")
xgb_data = loader.get_feature_matrix_for_xgboost(use_engineered=True)

X_train, y_train = xgb_data["X_train"], xgb_data["y_train"]
X_val, y_val = xgb_data["X_val"], xgb_data["y_val"]

# Combine train+val for tuning
X = np.vstack([X_train, X_val])
y = np.concatenate([y_train, y_val])

print("Data loaded:", X.shape)

# Class imbalance weight
scale_pos = (y == 0).sum() / (y == 1).sum()
print("Scale_pos_weight:", scale_pos)

# Define base XGBoost model
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=SEED,
    tree_method="hist"
)

# Parameter grid
param_dist = {
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "n_estimators": [200, 300, 400, 600],
    "scale_pos_weight": [scale_pos, scale_pos * 1.5, scale_pos * 2]
}

# Use AUC-PR because dataset is imbalanced
scorer = "average_precision"

search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=30,
    scoring=scorer,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=SEED
)

print("Running tuning...")
search.fit(X, y)

print("Best params:", search.best_params_)
print("Best AUC-PR:", search.best_score_)

# Save results
with open("results/xgb_tuning_results.json", "w") as f:
    json.dump({
        "best_params": search.best_params_,
        "best_score": search.best_score_
    }, f, indent=2)

print("Saved tuning results → results/xgb_tuning_results.json")
