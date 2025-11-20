import json
import os
import pandas as pd
import numpy as np

# Path to the JSON file
JSON_PATH = "results/full_experiments_20251117_231431.json"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = script_dir

    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Could not find JSON file at: {JSON_PATH}")

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # engineered_weight config
    engineered_cfg = None
    for cfg in data:
        if (
            cfg.get("config_name") == "engineered_weight"
            and cfg.get("use_engineered_features") is True
            and cfg.get("use_class_weights") is True
        ):
            engineered_cfg = cfg
            break

    if engineered_cfg is None:
        raise ValueError(
            "Could not find config with config_name='engineered_weight', "
            "use_engineered_features=True, use_class_weights=True"
        )

    models_dict = engineered_cfg["models"]

    metric_rows = []
    cm_rows = []

    # Collect metrics per model
    for model_name, metrics in models_dict.items():
        # tn, fp, fn, tp
        tn = metrics.get("tn")
        fp = metrics.get("fp")
        fn = metrics.get("fn")
        tp = metrics.get("tp")

        # If needed, derive from confusion_matrix
        if None in (tn, fp, fn, tp) and "confusion_matrix" in metrics:
            cm_raw = np.array(metrics["confusion_matrix"])
            # sklearn style: rows=actual [N,P], cols=pred [N,P]
            # [[TN, FP],
            #  [FN, TP]]
            tn, fp = cm_raw[0, 0], cm_raw[0, 1]
            fn, tp = cm_raw[1, 0], cm_raw[1, 1]

        # Build confusion matrix in your layout:
        # rows = Pred [P, N], cols = Actual [P, N]
        # [[TP, FP],
        #  [FN, TN]]
        cm_pred_actual = np.array([[tp, fp],
                                   [fn, tn]])

        print(f"\nConfusion matrix for {model_name} "
              "(rows=Pred [P,N], cols=Actual [P,N]):")
        print(cm_pred_actual)

        # Row for metrics summary
        metric_row = {
            "model": model_name,
            "precision": metrics.get("precision", None),
            "recall": metrics.get("recall", None),
            "f1": metrics.get("f1", None),
            "auc_roc": metrics.get("auc_roc", None),
            "auc_pr": metrics.get("auc_pr", None),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "best_val_f1": metrics.get("best_val_f1", None),
            "training_time": metrics.get("training_time", None),
        }
        metric_rows.append(metric_row)

        # Row for confusion-matrix-only table with your labels
        cm_rows.append(
            {
                "model": model_name,
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

    # Build metrics table, round & sort
    df_metrics = pd.DataFrame(metric_rows).set_index("model")

    # Round selected float columns to 4 decimal places
    float_cols = [
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "auc_pr",
        "best_val_f1",
        "training_time",
    ]
    for col in float_cols:
        if col in df_metrics.columns:
            df_metrics[col] = df_metrics[col].round(4)

    # Sort by highest recall, then highest AUC-PR
    df_metrics = df_metrics.sort_values(
        by=["recall", "auc_pr"],
        ascending=[False, False],
    )

    print("\n=== Engineered + Weights: metrics summary ===")
    print(df_metrics)

    metrics_csv = os.path.join(out_dir, "baseline_metrics_summary.csv")
    df_metrics.to_csv(metrics_csv)
    print(f"\nSaved metrics summary to {metrics_csv}")

    # Build confusion matrix table in your layout
    df_cm = pd.DataFrame(cm_rows).set_index("model")
    # reorder rows to match df_metrics order
    df_cm = df_cm.loc[df_metrics.index]

    print("\n=== Engineered + Weights: confusion matrix table "
          "(rows=Pred [P,N], cols=Actual [P,N]) ===")
    print(df_cm)

    cm_csv = os.path.join(out_dir, "baseline_confusion_matrices.csv")
    df_cm.to_csv(cm_csv)
    print(f"\nSaved confusion matrix table to {cm_csv}")


if __name__ == "__main__":
    main()