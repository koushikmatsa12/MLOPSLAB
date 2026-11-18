"""Task 3: Model Training, Comparison, Selection, and MLflow Model Registry.

Trains Decision Tree and Random Forest classifiers using reusable features from the Feature Store.
Compares Precision, Recall, F1-score, Accuracy, ROC-AUC, and Confusion Matrices.
Selects the model that better identifies possible water shortages (prioritizing Recall on the
positive shortage class, with F1-score as tie-breaker) and registers it in MLflow Model Registry.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    EVALUATION_RESULTS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_NUMERIC_FEATURES,
    MODELS_DIR,
    PRODUCTION_MODEL_PATH,
    TARGET_COLUMN,
)
from src.pipeline import load_from_feature_store


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn preprocessing transformer for numeric and categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), MODEL_NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                MODEL_CATEGORICAL_FEATURES,
            ),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute classification metrics and confusion matrix."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    return {
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def train_and_register_models() -> dict:
    """Train Decision Tree and Random Forest, log to MLflow, select best shortage detector, and register."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    feature_cols = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES

    # Load all feature store datasets
    df_v1 = load_from_feature_store("v1")
    df_v2 = load_from_feature_store("v2")
    df_combined = load_from_feature_store("combined")

    # Log baseline version-specific evaluation runs for traceability (v1 vs v2 drift tracking)
    version_drift_summary = {}
    for ver_name, df_ver in [("v1", df_v1), ("v2", df_v2)]:
        X_v = df_ver[feature_cols]
        y_v = df_ver[TARGET_COLUMN]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_v, y_v, test_size=0.2, random_state=42, stratify=y_v
        )
        rf_ver = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=7,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        rf_ver.fit(X_tr, y_tr)
        ver_metrics = evaluate_model(rf_ver, X_te, y_te)
        version_drift_summary[ver_name] = ver_metrics

        with mlflow.start_run(run_name=f"RandomForest_Dataset_{ver_name}"):
            mlflow.log_params(
                {
                    "model_type": "RandomForestClassifier",
                    "dataset_version": ver_name,
                    "n_samples": int(len(df_ver)),
                    "n_estimators": 150,
                    "max_depth": 7,
                    "class_weight": "balanced",
                }
            )
            mlflow.log_metrics(
                {
                    "precision": ver_metrics["precision"],
                    "recall": ver_metrics["recall"],
                    "f1_score": ver_metrics["f1_score"],
                    "accuracy": ver_metrics["accuracy"],
                    "roc_auc": ver_metrics["roc_auc"],
                }
            )

    # Primary comparison & model selection on combined v1 + v2 Feature Store (500 nights)
    # so the production model handles both normal and summer (+35% demand) regimes accurately.
    X = df_combined[feature_cols]
    y = df_combined[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    candidate_specs = {
        "DecisionTree": {
            "estimator": DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42,
            ),
            "params": {
                "model_type": "DecisionTreeClassifier",
                "dataset_version": "combined_v1_v2",
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "max_depth": 5,
                "min_samples_split": 8,
                "min_samples_leaf": 4,
            },
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            ),
            "params": {
                "model_type": "RandomForestClassifier",
                "dataset_version": "combined_v1_v2",
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
            },
        },
    }

    trained_models: dict[str, Pipeline] = {}
    comparison_results: dict[str, dict] = {}
    run_ids: dict[str, str] = {}

    for model_name, spec in candidate_specs.items():
        pipe = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("classifier", spec["estimator"]),
            ]
        )
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)

        # Also check subgroup recall on v1 (normal) and v2 (summer) test records
        test_meta = df_combined.loc[X_test.index, "dataset_version"]
        v1_mask = test_meta == "v1"
        v2_mask = test_meta == "v2"
        metrics["v1_test_recall"] = round(
            float(recall_score(y_test[v1_mask], pipe.predict(X_test[v1_mask]), zero_division=0)),
            4,
        )
        metrics["v2_test_recall"] = round(
            float(recall_score(y_test[v2_mask], pipe.predict(X_test[v2_mask]), zero_division=0)),
            4,
        )

        # Extract feature importances aligned with transformed column names
        ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(MODEL_CATEGORICAL_FEATURES))
        all_feat_names = MODEL_NUMERIC_FEATURES + cat_names
        importances = pipe.named_steps["classifier"].feature_importances_
        feat_imp = {
            name: round(float(val), 4)
            for name, val in sorted(
                zip(all_feat_names, importances), key=lambda kv: kv[1], reverse=True
            )
        }
        metrics["feature_importances"] = feat_imp

        with mlflow.start_run(run_name=f"{model_name}_Comparison") as run:
            mlflow.log_params(spec["params"])
            mlflow.log_metrics(
                {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "accuracy": metrics["accuracy"],
                    "roc_auc": metrics["roc_auc"],
                    "true_negatives": metrics["tn"],
                    "false_positives": metrics["fp"],
                    "false_negatives": metrics["fn"],
                    "true_positives": metrics["tp"],
                    "v1_test_recall": metrics["v1_test_recall"],
                    "v2_test_recall": metrics["v2_test_recall"],
                }
            )
            mlflow.log_dict(
                {
                    "confusion_matrix": metrics["confusion_matrix"],
                    "labels": ["Adequate Reserve (0)", "Shortage Risk (1)"],
                    "feature_importances": feat_imp,
                },
                "evaluation_details.json",
            )
            log_model_kwargs = {
                "sk_model": pipe,
                "name": "model",
                "input_example": X_train.head(3).astype(
                    {col: "float64" for col in MODEL_NUMERIC_FEATURES}
                ),
            }
            if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
                log_model_kwargs["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
            mlflow.sklearn.log_model(**log_model_kwargs)
            run_ids[model_name] = run.info.run_id

        trained_models[model_name] = pipe
        comparison_results[model_name] = metrics

    # Select the model that better identifies possible shortages:
    # Prioritize Recall (minimizing missed shortages / False Negatives), then F1-score.
    selected_name = max(
        comparison_results.keys(),
        key=lambda m: (
            comparison_results[m]["recall"],
            comparison_results[m]["f1_score"],
            comparison_results[m]["precision"],
        ),
    )
    selected_run_id = run_ids[selected_name]
    selected_model = trained_models[selected_name]
    selected_metrics = comparison_results[selected_name]

    # Register selected model in MLflow Model Registry
    model_uri = f"runs:/{selected_run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)

    # Set alias 'production' on the registered model version
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias="production",
        version=registered.version,
    )
    client.update_model_version(
        name=MLFLOW_MODEL_NAME,
        version=registered.version,
        description=(
            f"Selected {selected_name} classifier for overnight residential water shortage "
            f"prediction. Recall={selected_metrics['recall']}, "
            f"Precision={selected_metrics['precision']}, F1={selected_metrics['f1_score']}."
        ),
    )

    # Save local production artifact for fast FastAPI serving and portable CI testing
    joblib.dump(selected_model, PRODUCTION_MODEL_PATH)

    model_metadata = {
        "selected_model": selected_name,
        "selection_criterion": "Highest Shortage Recall (minimizing False Negatives), tie-broken by F1-score",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "mlflow_experiment_name": MLFLOW_EXPERIMENT_NAME,
        "mlflow_registered_model": MLFLOW_MODEL_NAME,
        "mlflow_model_version": str(registered.version),
        "mlflow_alias": "production",
        "mlflow_run_id": selected_run_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            k: v for k, v in selected_metrics.items() if k != "feature_importances"
        },
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    evaluation_payload = {
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
        "selected_model": selected_name,
        "registered_model_name": MLFLOW_MODEL_NAME,
        "registered_model_version": str(registered.version),
        "run_ids": run_ids,
        "test_set_size": int(len(X_test)),
        "models": comparison_results,
        "version_drift_evaluation": version_drift_summary,
    }
    with open(EVALUATION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=2)

    return evaluation_payload


if __name__ == "__main__":
    results = train_and_register_models()
    print(f"Selected & Registered Model: {results['selected_model']}")
    for name, m in results["models"].items():
        print(
            f"  {name}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, "
            f"F1={m['f1_score']:.4f}, Accuracy={m['accuracy']:.4f}, CM={m['confusion_matrix']}"
        )

