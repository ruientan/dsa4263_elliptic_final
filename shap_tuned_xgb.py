
import os
import numpy as np
import shap
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from data_loader import EllipticDataLoader
from feature_eng import FeatureEngineer

# -------------------------------------------------------------------
# 1. Load tuned model
# -------------------------------------------------------------------
MODEL_PATH = "models/xgboost_engineered_weight_tuned.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find {MODEL_PATH}. "
        "Make sure you've trained and saved the tuned XGBoost model first."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Loaded tuned model from:", MODEL_PATH)

# -------------------------------------------------------------------
# 2. Recreate engineered feature matrices (same pipeline as training)
# -------------------------------------------------------------------
loader = EllipticDataLoader(data_dir="data")
xgb_data = loader.get_feature_matrix_for_xgboost(use_engineered=True)

X_train = xgb_data["X_train"]
y_train = xgb_data["y_train"]
X_test  = xgb_data["X_test"]
y_test  = xgb_data["y_test"]

print("X_train shape:", X_train.shape)
print("X_test  shape:", X_test.shape)

# -------------------------------------------------------------------
# 3. Recover feature names (baseline + engineered)
# -------------------------------------------------------------------
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

# Optional consistency check
assert X_train.shape[1] == len(feature_names), "Feature count mismatch!"

# -------------------------------------------------------------------
# 4. Build SHAP explainer – tuned XGBoost
# -------------------------------------------------------------------
background_size = min(10000, X_train.shape[0])
background_idx = np.random.choice(X_train.shape[0], size=background_size, replace=False)
background = X_train[background_idx]

print("Background size for SHAP:", background.shape[0])

explainer = shap.TreeExplainer(model, data=background)

X_test_shap = X_test
y_test_shap = y_test

print("Computing SHAP values on", X_test_shap.shape[0], "test samples...")

shap_values = explainer.shap_values(X_test_shap)

if isinstance(shap_values, list):
    shap_vals = shap_values[0]
else:
    shap_vals = shap_values

print("SHAP values shape:", np.array(shap_vals).shape)

# -------------------------------------------------------------------
# 5. Save global SHAP plots (top 30 features for visualisation)
# -------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

# 5a. Beeswarm summary
plt.figure()
shap.summary_plot(
    shap_vals,
    X_test_shap,
    feature_names=feature_names,
    max_display=30,
    show=False
)
plt.tight_layout()
plt.savefig("results/shap_summary_xgb_tuned_engineered_weight.png",
            dpi=300, bbox_inches="tight")
plt.close()

print("Saved SHAP summary plot.")

# 5b. Bar plot
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
plt.savefig("results/shap_bar_xgb_tuned_engineered_weight.png",
            dpi=300, bbox_inches="tight")
plt.close()

print("Saved SHAP bar plot.")

# -------------------------------------------------------------------
# 6. SHAP feature importance + cumulative 95% selection
# -------------------------------------------------------------------
print("\nExtracting SHAP feature importances...")

# Mean |SHAP| per feature across test samples
mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

# ---- 6a. Save full ranking (for reference)
full_csv = "results/shap_feature_ranking_xgb_tuned.csv"
feature_importance_df.to_csv(full_csv, index=False)
print(f"Saved full SHAP ranking → {full_csv}")

# ---- 6b. Top / bottom 30 (still useful to eyeball)
top_30 = feature_importance_df.head(30)
top_csv = "results/shap_top30_xgb_tuned.csv"
top_30.to_csv(top_csv, index=False)
print(f"Saved TOP 30 features → {top_csv}")

bottom_30 = feature_importance_df.tail(30)
bottom_csv = "results/shap_bottom30_xgb_tuned.csv"
bottom_30.to_csv(bottom_csv, index=False)
print(f"Saved BOTTOM 30 features → {bottom_csv}")

print("\nTop 10 most important features:")
print(top_30.head(10).to_string(index=False))

print("\nBottom 10 least important features:")
print(bottom_30.tail(10).to_string(index=False))

# ---- 6c. Method 2: cumulative 95% SHAP importance
print("\nComputing cumulative SHAP importance (95% threshold)...")

total_importance = feature_importance_df["mean_abs_shap"].sum()
# Avoid division by zero if something weird happens
if total_importance == 0:
    raise ValueError("Total SHAP importance is zero – check shap_vals or model output.")

feature_importance_df["rel_importance"] = (
    feature_importance_df["mean_abs_shap"] / total_importance
)
feature_importance_df["cumulative_importance"] = (
    feature_importance_df["rel_importance"].cumsum()
)

THRESHOLD = 0.95
selected_df = feature_importance_df[
    feature_importance_df["cumulative_importance"] <= THRESHOLD
].copy()

# Safety: ensure we at least keep one feature
if selected_df.empty:
    selected_df = feature_importance_df.iloc[[0]].copy()

selected_features = selected_df["feature"].tolist()

print(f"\nSelected {len(selected_features)} features that explain "
      f"{THRESHOLD*100:.1f}% of total |SHAP|.")

sel_csv = "results/shap_selected95_xgb_tuned.csv"
selected_df.to_csv(sel_csv, index=False)
print(f"Saved 95%-cumulative SHAP feature set → {sel_csv}")

print("\nFirst 10 selected features (by importance):")
print(selected_df.head(10).to_string(index=False))

print("\nLast 10 selected features (closest to 95% cutoff):")
print(selected_df.tail(10).to_string(index=False))

print("\nDone. Check all results inside the results/ folder.")
