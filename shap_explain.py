import os
import numpy as np
import shap
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from data_loader import EllipticDataLoader
from feature_eng import FeatureEngineer

# Load the best model (XGBoost enginnered_weight)
MODEL_PATH = "models/engineered_weight_xgboost_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find {MODEL_PATH}. "
        "Make sure you've run the experiment suite with engineered_weight + xgboost first."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Loaded model from:", MODEL_PATH)

# Recreate engineered feature matrix (same as training)
loader = EllipticDataLoader(data_dir="data")

# This gives you the *scaled* numpy matrices used for training/testing
xgb_data = loader.get_feature_matrix_for_xgboost(use_engineered=True)

X_train = xgb_data["X_train"]
y_train = xgb_data["y_train"]
X_test  = xgb_data["X_test"]
y_test  = xgb_data["y_test"]

print("X_train shape:", X_train.shape)
print("X_test  shape:", X_test.shape)

# Recover feature names
tx_features, tx_classes, _, addr_tx_in, tx_addr_out = loader.load_raw_data()
engineer = FeatureEngineer("data")
df_eng, eng_features = engineer.engineer_features(
    tx_features, tx_classes, addr_tx_in, tx_addr_out
)

exclude_cols = ['txId', 'Time step', 'time_step', 'class', 'label', 'out_range_ratio']
feature_cols = [c for c in df_eng.columns if c not in exclude_cols]

# Only numeric columns, in the same order
numeric_feature_cols = df_eng[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
feature_names = numeric_feature_cols

print("Number of engineered+baseline features:", len(feature_names))




# 4. Build SHAP explainer
# Use a subset of training data as background (for speed)

background_size = min(10000, X_train.shape[0])
background_idx = np.random.choice(X_train.shape[0], size=background_size, replace=False)
background = X_train[background_idx]

print("Background size for SHAP:", background.shape[0])

explainer = shap.TreeExplainer(model, data=background)

# For SHAP plots we can also subsample test set if it's very big
X_test_shap = X_test
y_test_shap = y_test

print("Computing SHAP values on", X_test_shap.shape[0], "test samples...")

# Compute SHAP values
shap_values = explainer.shap_values(X_test_shap)

# XGBoost binary classifier usually returns an array, but guard for list case
if isinstance(shap_values, list):
    shap_vals = shap_values[0]
else:
    shap_vals = shap_values

print("SHAP values shape:", np.array(shap_vals).shape)

# -------------------------------------------------------------------
# 5. Save global SHAP plots
# -------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

# Summary dot plot (beeswarm)
plt.figure()
shap.summary_plot(
    shap_vals,
    X_test_shap,
    feature_names=feature_names,
    max_display=30,
    show=False
)
plt.tight_layout()
plt.savefig("results/shap_summary_engineered_weight.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved SHAP summary plot to results/shap_summary_engineered_weight.png")

# Bar plot of mean |SHAP|
plt.figure()
shap.summary_plot(
    shap_vals,
    X_test_shap,
    feature_names=feature_names,
    plot_type="bar",
    max_display=30,
    show=False
)
plt.tight_layout()
plt.savefig("results/shap_bar_engineered_weight.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved SHAP bar plot to results/shap_bar_engineered_weight.png")

print("\nDone. You can now inspect the PNG files in the results/ folder.")