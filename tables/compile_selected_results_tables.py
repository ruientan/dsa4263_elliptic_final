import json
import os
import pandas as pd
import numpy as np

MODEL_RESULTS = [
    # 1. GCN baseline_weight (no engineered features, class weights = True)
    {
        "name": "gcn_baseline_weight",
        "path": "results/full_experiments_20251120_142248.json",
        "metrics_key": None,
        "config_name": "baseline_weight",
        "model_key": "gcn"
    },

    # 2. GCN engineered_weight (engineered features + class weights = True)
    {
        "name": "gcn_engineered_weight",
        "path": "results/full_experiments_20251120_142248.json",
        "metrics_key": None,
        "config_name": "engineered_weight",
        "model_key": "gcn"
    },

    # 3. GraphSAGE engineered_weight (engineered features + class weights = True)
    {
        "name": "graphsage_engineered_weight",
        "path": "results/full_experiments_20251120_142248.json",
        "metrics_key": None,
        "config_name": "engineered_weight",
        "model_key": "graphsage"
    },

    # 4. XGB engineered_weight (engineered features + class weights = True)
    {
        "name": "xgb_engineered_weight",
        "path": "results/full_experiments_20251120_142248.json",
        "metrics_key": None,
        "config_name": "engineered_weight",
        "model_key": "xgboost"
    },

    # 5. Ensemble: GraphSAGE + XGB (engineered features + class weights = True)
    {
        "name": "ensemble_graphsage_xgb_engineered_weight",
        "path": "results/ensemble_graphsage_xgb_test_metrics.json",
        "metrics_key": None,
        "config_name": None,
        "model_key": None
    }
]


# Helper to load metrics from JSON (handles nested vs flat)
def load_metrics_from_json(path, metrics_key=None, config_name=None, model_key=None):
    with open(path, "r") as f:
        data = json.load(f)

    # Multi-config JSON (like the one you pasted)
    if config_name is not None and model_key is not None:
        # data is a list of configs
        for item in data:
            if item.get("config_name") == config_name:
                cfg = item
                break
        else:
            raise ValueError(f"config_name='{config_name}' not found in {path}")

        models_dict = cfg.get("models", {})
        if model_key not in models_dict:
            raise KeyError(
                f"model_key='{model_key}' not found under config '{config_name}'. "
                f"Available: {list(models_dict.keys())}"
            )
        metrics = models_dict[model_key]

    # Simple JSON (flat or nested)
    else:
        if metrics_key is not None:
            metrics = data[metrics_key]
        else:
            metrics = data

    return metrics


# Main aggregation logic
def main():
    # Save tables into the same folder as this script (tables/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = script_dir

    metric_rows = []
    cm_rows = []

    for cfg in MODEL_RESULTS:
        name = cfg["name"]
        path = cfg["path"]
        metrics_key = cfg.get("metrics_key")
        config_name = cfg.get("config_name")
        model_key = cfg.get("model_key")

        print(f"Loading metrics for {name} from {path}...")
        metrics = load_metrics_from_json(
            path,
            metrics_key=metrics_key,
            config_name=config_name,
            model_key=model_key,
        )

        # ----- tn, fp, fn, tp -----
        tn = metrics.get("tn")
        fp = metrics.get("fp")
        fn = metrics.get("fn")
        tp = metrics.get("tp")

        # If tn/fp/fn/tp not all present but confusion_matrix exists, derive them
        if None in (tn, fp, fn, tp) and "confusion_matrix" in metrics:
            cm_raw = np.array(metrics["confusion_matrix"])
            # sklearn style: rows = actual [N,P], cols = pred [N,P]
            # [[TN, FP],
            #  [FN, TP]]
            tn, fp = cm_raw[0, 0], cm_raw[0, 1]
            fn, tp = cm_raw[1, 0], cm_raw[1, 1]

        # Build confusion matrix in YOUR layout:
        # rows = Pred [P, N], cols = Actual [P, N]
        # [[TP, FP],
        #  [FN, TN]]
        cm_pred_actual = np.array([[tp, fp],
                                   [fn, tn]])

        print(
            f"\nConfusion matrix for {name} "
            "(rows=Pred [P,N], cols=Actual [P,N]):"
        )
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

        # Include optional fields if present
        if "best_val_f1" in metrics:
            metric_row["best_val_f1"] = metrics["best_val_f1"]
        if "training_time" in metrics:
            metric_row["training_time"] = metrics["training_time"]

        metric_rows.append(metric_row)

        # Row for confusion-matrix-only table using your labels
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


    # Build metrics table, round & sort (recall desc, then auc_pr desc)
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
    if "recall" in df_metrics.columns and "auc_pr" in df_metrics.columns:
        df_metrics = df_metrics.sort_values(
            by=["recall", "auc_pr"],
            ascending=[False, False],
        )

    print("\n=== Selected models: metrics summary ===")
    print(df_metrics)

    metrics_csv = os.path.join(out_dir, "selected_models_metrics_summary.csv")
    df_metrics.to_csv(metrics_csv)
    print(f"\nSaved metrics summary to {metrics_csv}")


    # Build confusion matrix table in your layout
    df_cm = pd.DataFrame(cm_rows).set_index("model")
    # Reorder rows to match df_metrics ordering
    df_cm = df_cm.loc[df_metrics.index]

    print(
        "\n=== Selected models: confusion matrix table "
        "(rows=Pred [P,N], cols=Actual [P,N]) ==="
    )
    print(df_cm)

    cm_csv = os.path.join(out_dir, "selected_models_confusion_matrices.csv")
    df_cm.to_csv(cm_csv)
    print(f"\nSaved confusion matrix table to {cm_csv}")


if __name__ == "__main__":
    main()