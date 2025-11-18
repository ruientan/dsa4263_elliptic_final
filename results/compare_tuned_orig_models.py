import os
import pandas as pd

os.makedirs("results", exist_ok=True)

# ===============================
# ORIGINAL MODEL RESULTS
# ===============================

orig_xgb = {
    "Model": "XGBoost (Original)",
    "Precision": 0.7398,
    "Recall": 0.4877,
    "F1": 0.5879,
    "AUC_PR": 0.5617,
    "AUC_ROC": 0.8544,
    "TP": 199, "FP": 70, "FN": 209, "TN": 8363,
}

orig_sage = {
    "Model": "GraphSAGE (Original)",
    "Precision": 0.5000,
    "Recall": 0.4657,
    "F1": 0.4822,
    "AUC_PR": 0.4211,
    "AUC_ROC": 0.8284,
    "TP": 190, "FP": 190, "FN": 218, "TN": 8243,
}

orig_gcn = {
    "Model": "GCN (Original)",
    "Precision": 0.3479,
    "Recall": 0.4485,
    "F1": 0.3919,
    "AUC_PR": 0.3806,
    "AUC_ROC": 0.8421,
    "TP": 183, "FP": 343, "FN": 225, "TN": 8090,
}

# ===============================
# TUNED MODELS
# ===============================

tuned_xgb = {
    "Model": "XGBoost (Tuned)",
    "Precision": 0.9346,
    "Recall": 0.4902,
    "F1": 0.6431,
    "AUC_PR": 0.6837,
    "AUC_ROC": 0.9442,
    "TP": 200, "FP": 14, "FN": 208, "TN": 8419,
}

tuned_sage = {
    "Model": "GraphSAGE (Tuned)",
    "Precision": 0.6246,
    "Recall": 0.4485,
    "F1": 0.5221,
    "AUC_PR": 0.4540,
    "AUC_ROC": 0.8296,
    "TP": 183, "FP": 110, "FN": 225, "TN": 8323,
}

tuned_gcn = {
    "Model": "GCN (Tuned)",
    "Precision": 0.4215,
    "Recall": 0.4412,
    "F1": 0.4311,
    "AUC_PR": 0.4410,
    "AUC_ROC": 0.8571,
    "TP": 180, "FP": 247, "FN": 228, "TN": 8186,
}

# ===============================
# Build the comparison table
# ===============================

df = pd.DataFrame([
    orig_xgb, tuned_xgb,
    orig_sage, tuned_sage,
    orig_gcn, tuned_gcn
])

# Sort using your project's priority:
# 1) Recall → 2) AUC-PR → 3) F1
df_sorted = df.sort_values(["Recall", "AUC_PR", "F1"], ascending=False)

print("\n=== TUNED AND ORIGINAL MODEL COMPARISON TABLE ===\n")
print(df_sorted.to_string(index=False))

# Save CSV
out_path = "results/tuned_orig_model_comparison.csv"
df_sorted.to_csv(out_path, index=False)

print(f"\nSaved comparison CSV to: {out_path}")
