import os
import json
import pandas as pd

os.makedirs("results", exist_ok=True)


orig_xgb = {
    "Model": "XGBoost (Original)",
    "Precision": 0.739777,
    "Recall": 0.487745,
    "F1": 0.587888,
    "AUC_PR": 0.561737,
    "AUC_ROC": 0.854413,
    "TP": 199,
    "FP": 70,
    "FN": 209,
    "TN": 8363,
}

orig_sage = {
    "Model": "GraphSAGE (Original)",
    "Precision": 0.702041,
    "Recall": 0.421569,
    "F1": 0.526799,
    "AUC_PR": 0.487775,
    "AUC_ROC": 0.828845,
    "TP": 172,
    "FP": 73,
    "FN": 236,
    "TN": 8360,
}

orig_gcn = {
    "Model": "GCN (Original)",
    "Precision": 0.376682,
    "Recall": 0.411765,
    "F1": 0.393443,
    "AUC_PR": 0.399067,
    "AUC_ROC": 0.829381,
    "TP": 168,
    "FP": 278,
    "FN": 240,
    "TN": 8155,
}

orig_gcn_bw = {
    "Model": "GCN (Baseline Weight)",
    "Precision": 0.600000,
    "Recall": 0.345588,
    "F1": 0.438569,
    "AUC_PR": 0.445448,
    "AUC_ROC": 0.824181,
    "TP": 141,
    "FP": 94,
    "FN": 267,
    "TN": 8339,
}

# ===============================
# TUNED XGBOOST (engineered_weight)
# Read from JSON: xgboost_engineered_weight_tuned_test_metrics.json
# ===============================

tuned_json_path = "results/xgb_engineered_weight_tuned_test_metrics.json"
if not os.path.exists(tuned_json_path):
    raise FileNotFoundError(
        f"Could not find {tuned_json_path}. "
        "Make sure you've run train_tuned_xgboost.py first."
    )

with open(tuned_json_path, "r") as f:
    tuned_metrics = json.load(f)

tuned_xgb = {
    "Model": "XGBoost (Tuned)",
    "Precision": tuned_metrics["precision"],
    "Recall": tuned_metrics["recall"],
    "F1": tuned_metrics["f1"],
    "AUC_PR": tuned_metrics["auc_pr"],
    "AUC_ROC": tuned_metrics["auc_roc"],
    "TP": tuned_metrics["tp"],
    "FP": tuned_metrics["fp"],
    "FN": tuned_metrics["fn"],
    "TN": tuned_metrics["tn"],
}

# ===============================
# Build the comparison table
# ===============================

df = pd.DataFrame([
    orig_xgb,
    tuned_xgb,
    orig_sage,
    orig_gcn,
    orig_gcn_bw,
])

# Priority: Recall → AUC-PR → F1 (descending)
df_sorted = df.sort_values(["Recall", "AUC_PR", "F1"], ascending=False)

print("\n=== TUNED VS ORIGINAL MODEL COMPARISON (engineered_weight) ===\n")
print(df_sorted.to_string(index=False))

# Save CSV
out_path = "results/tuned_orig_model_comparison.csv"
df_sorted.to_csv(out_path, index=False)

print(f"\nSaved comparison CSV to: {out_path}")
