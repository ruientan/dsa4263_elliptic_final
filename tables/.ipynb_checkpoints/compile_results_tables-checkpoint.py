import json
import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# 1. Configure your result files here
#    - name: label for the model in the table
#    - path: path to the JSON file (relative to project root)
#    - metrics_key: key where metrics live if nested, else None
# ------------------------------------------------------------------------------

MODEL_RESULTS = [
    # Finetuned
    {
        "name": "gcn_tuned",
        "path": "results/gcn_tuning_results.json",
        "metrics_key": "best_test_metrics",
    },
    {
        "name": "xgboost_tuned",
        "path": "results/xgb_engineered_weight_tuned_test_metrics.json",
        "metrics_key": None,
    },
    {
        "name": "graphsage_tuned",
        "path": "results/graphsage_tuning_results.json",
        "metrics_key": "best_test_metrics",
    },
    {
        "name": "ensemble_graphsage_xgb",
        "path": "results/ensemble_graphsage_xgb_test_metrics.json",
        "metrics_key": None,
    },
    # add more if needed
]


# ------------------------------------------------------------------------------
# 2. Helper to load metrics from JSON (handles nested vs flat)
# ------------------------------------------------------------------------------

def load_metrics_from_json(path, metrics_key=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if metrics_key is not None:
        if metrics_key not in data:
            raise KeyError(
                f"Expected key '{metrics_key}' in {path}, found keys: {list(data.keys())}"
            )
        metrics = data[metrics_key]
    else:
        metrics = data

    return metrics


# ------------------------------------------------------------------------------
# 3. Main aggregation logic
# ------------------------------------------------------------------------------

def main():
    # Ensure outputs go into the same 'tables' folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = script_dir  # this is the 'tables' folder

    metric_rows = []
    cm_rows = []

    for cfg in MODEL_RESULTS:
        name = cfg["name"]
        path = cfg["path"]
        metrics_key = cfg["metrics_key"]

        print(f"Loading metrics for {name} from {path}...")
        metrics = load_metrics_from_json(path, metrics_key=metrics_key)

        # --- Get tn, fp, fn, tp (from confusion_matrix or individual keys) ---
        tn = metrics.get("tn")
        fp = metrics.get("fp")
        fn = metrics.get("fn")
        tp = metrics.get("tp")

        # If tn/fp/fn/tp are missing but confusion_matrix exists, derive them
        if None in (tn, fp, fn, tp) and "confusion_matrix" in metrics:
            cm_raw = np.array(metrics["confusion_matrix"])
            # sklearn layout: rows = actual [N,P], cols = predicted [N,P]
            # [[TN, FP],
            #  [FN, TP]]
            tn, fp = cm_raw[0, 0], cm_raw[0, 1]
            fn, tp = cm_raw[1, 0], cm_raw[1, 1]

        # --- Build confusion matrix in YOUR desired layout ---
        # Rows = Predicted [P, N], Cols = Actual [P, N]
        # [[TP, FP],
        #  [FN, TN]]
        cm_pred_actual = np.array([[tp, fp],
                                   [fn, tn]])

        print(f"\nConfusion matrix for {name} "
              "(rows=Pred [P,N], cols=Actual [P,N]):")
        print(cm_pred_actual)

        # Row for metrics summary table
        metric_row = {
            "model": name,
            "precision": metrics.get("precision", None),
            "recall": metrics.get("recall", None),
            "f1": metrics.get("f1", None),
            "auc_roc": metrics.get("auc_roc", None),
            "auc_pr": metrics.get("auc_pr", None),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

        # Include best_val_f1 if present (e.g. GraphSAGE tuning)
        if "best_val_f1" in metrics:
            metric_row["best_val_f1"] = metrics["best_val_f1"]

        metric_rows.append(metric_row)

        # Row for confusion-matrix-only table using your layout labels
        cm_rows.append(
            {
                "model": name,
                # Predicted P, Actual P = TP
                "PredP_ActP_TP": tp,
                # Predicted P, Actual N = FP
                "PredP_ActN_FP": fp,
                # Predicted N, Actual P = FN
                "PredN_ActP_FN": fn,
                # Predicted N, Actual N = TN
                "PredN_ActN_TN": tn,
            }
        )

    # --- Build and save summary metrics table ---
    df_metrics = pd.DataFrame(metric_rows).set_index("model")

    # Round metric columns to 4 decimal places
    float_cols = ["precision", "recall", "f1", "auc_roc", "auc_pr", "best_val_f1"]
    for col in float_cols:
        if col in df_metrics.columns:
            df_metrics[col] = df_metrics[col].round(4)

    # Sort rows: highest recall first, then highest auc_pr
    df_metrics = df_metrics.sort_values(by=["recall", "auc_pr"],
                                        ascending=[False, False])
        
    print("\n=== Summary table of evaluation metrics ===")
    print(df_metrics)

    metrics_csv = os.path.join(out_dir, "all_models_metrics_summary.csv")
    df_metrics.to_csv(metrics_csv)
    print(f"\nSaved metrics summary table to {metrics_csv}")

    # --- Build and save confusion matrix table in your layout ---
    df_cm = pd.DataFrame(cm_rows).set_index("model")
    df_cm = df_cm.loc[df_metrics.index]  # match metrics table order
    
    print("\n=== Confusion matrix table "
          "(rows=Pred [P,N], cols=Actual [P,N]) ===")
    print(df_cm)

    cm_csv = os.path.join(out_dir, "all_models_confusion_matrices.csv")
    df_cm.to_csv(cm_csv)
    print(f"\nSaved confusion matrix table to {cm_csv}")


if __name__ == "__main__":
    main()


# import json
# import os
# import pandas as pd
# import numpy as np

# # ------------------------------------------------------------------------------
# # 1. Configure your result files here
# #    - name: label for the model in the table
# #    - path: path to the JSON file (relative to project root)
# #    - metrics_key: key where metrics live if nested, else None
# # ------------------------------------------------------------------------------

# MODEL_RESULTS = [
#     {
#         "name": "gcn_tuned",
#         "path": "results/gcn_tuning_results.json",
#         "metrics_key": "best_test_metrics",
#     },
#     {
#         "name": "xgboost_tuned",
#         "path": "results/xgb_engineered_weight_tuned_test_metrics.json",
#         "metrics_key": None,
#     },
#     {
#         "name": "graphsage_tuned",
#         "path": "results/graphsage_tuning_results.json",
#         "metrics_key": "best_test_metrics",
#     },
#     {
#         "name": "ensemble_graphsage_xgb",
#         "path": "results/ensemble_graphsage_xgb_test_metrics.json",
#         "metrics_key": None,
#     },
    
# ]


# # ------------------------------------------------------------------------------
# # 2. Helper to load metrics from JSON (handles nested vs flat)
# # ------------------------------------------------------------------------------

# def load_metrics_from_json(path, metrics_key=None):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Metrics file not found: {path}")

#     with open(path, "r") as f:
#         data = json.load(f)

#     if metrics_key is not None:
#         if metrics_key not in data:
#             raise KeyError(
#                 f"Expected key '{metrics_key}' in {path}, found keys: {list(data.keys())}"
#             )
#         metrics = data[metrics_key]
#     else:
#         metrics = data

#     return metrics


# # ------------------------------------------------------------------------------
# # 3. Main aggregation logic
# # ------------------------------------------------------------------------------

# def main():
#     # Ensure outputs go into the same 'tables' folder as this script
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     out_dir = script_dir  # this is the 'tables' folder

#     rows = []
#     cm_rows = []

#     for cfg in MODEL_RESULTS:
#         name = cfg["name"]
#         path = cfg["path"]
#         metrics_key = cfg["metrics_key"]

#         print(f"Loading metrics for {name} from {path}...")
#         metrics = load_metrics_from_json(path, metrics_key=metrics_key)

#         # --- Confusion matrix handling ---
#         if "confusion_matrix" in metrics:
#             cm = np.array(metrics["confusion_matrix"])
#             # assume 2x2: [[tn, fp], [fn, tp]]
#             tn, fp = cm[0, 0], cm[0, 1]
#             fn, tp = cm[1, 0], cm[1, 1]
#         else:
#             tn = metrics.get("tn")
#             fp = metrics.get("fp")
#             fn = metrics.get("fn")
#             tp = metrics.get("tp")
#             if None not in (tn, fp, fn, tp):
#                 cm = np.array([[tn, fp], [fn, tp]])
#             else:
#                 cm = None

#         # row for metrics summary
#         row = {
#             "model": name,
#             "precision": metrics.get("precision", None),
#             "recall": metrics.get("recall", None),
#             "f1": metrics.get("f1", None),
#             "auc_roc": metrics.get("auc_roc", None),
#             "auc_pr": metrics.get("auc_pr", None),
#             "tn": tn,
#             "fp": fp,
#             "fn": fn,
#             "tp": tp,
#         }

#         # include best_val_f1 if present (e.g. GraphSAGE tuning)
#         if "best_val_f1" in metrics:
#             row["best_val_f1"] = metrics["best_val_f1"]

#         rows.append(row)

#         # row for confusion-matrix-only table
#         cm_rows.append(
#             {
#                 "model": name,
#                 "tn": tn,
#                 "fp": fp,
#                 "fn": fn,
#                 "tp": tp,
#             }
#         )

#     # --- Build and save summary metrics table ---
#     df_metrics = pd.DataFrame(rows).set_index("model")
#     print("\n=== Summary table of evaluation metrics ===")
#     print(df_metrics)

#     metrics_csv = os.path.join(out_dir, "all_models_metrics_summary.csv")
#     df_metrics.to_csv(metrics_csv)
#     print(f"\nSaved metrics summary table to {metrics_csv}")

#     # --- Build and save confusion matrix table ---
#     df_cm = pd.DataFrame(cm_rows).set_index("model")
#     print("\n=== Confusion matrix table (tn, fp, fn, tp) ===")
#     print(df_cm)

#     cm_csv = os.path.join(out_dir, "all_models_confusion_matrices.csv")
#     df_cm.to_csv(cm_csv)
#     print(f"\nSaved confusion matrix table to {cm_csv}")


# if __name__ == "__main__":
#     main()
