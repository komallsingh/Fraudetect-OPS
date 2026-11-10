
import os
import json
import joblib
import argparse
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from pathlib import Path
from mlflow import MlflowClient
from mlflow.models import infer_signature

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ---------------- CONFIGURATION ----------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

MODEL_NAME = "FraudGuard"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

RANDOM_STATE = 42


# ---------------- LOAD DATA ----------------

def load_data():

    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    X_train = train.drop(columns=["Class"])
    y_train = train["Class"]

    X_val = validation.drop(columns=["Class"])
    y_val = validation["Class"]

    X_test = test.drop(columns=["Class"])
    y_test = test["Class"]

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Training and validation features differ")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test features differ")

    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Both classes must exist in train and validation")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------- BUILD MODELS ----------------

def build_models():

    models = {

        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE
            ))
        ]),

        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

    }

    return models


# ---------------- THRESHOLD SELECTION ----------------

def select_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    if len(thresholds) == 0:
        raise ValueError("Unable to select classification threshold")

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = int(np.argmax(f1_scores))

    return float(thresholds[best_index])


# ---------------- EVALUATION ----------------

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),

        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),

        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),

        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),

        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        )
    }


# ---------------- MLflow INITIALIZATION ----------------

def initialize_mlflow():

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment("FraudGuard-Experiments")


# ---------------- TRAIN AND TRACK ----------------

def train_models(X_train, y_train, X_val, y_val):

    models = build_models()

    results = {}

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        with mlflow.start_run(run_name=model_name) as run:

            model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_val)[:, 1]

            threshold = select_threshold(
                y_val,
                probabilities
            )

            metrics = calculate_metrics(
                y_val,
                probabilities,
                threshold
            )

            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("classification_threshold", threshold)

            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("validation_rows", len(X_val))

            params = model.get_params()

            mlflow.log_params({
                f"model_{key}": str(value)
                for key, value in params.items()
                if isinstance(value, (str, int, float, bool))
                or value is None
            })

            # Log validation metrics
            mlflow.log_metrics(metrics)

            mlflow.set_tag(
                "selection_metric",
                "validation_pr_auc"
            )

            mlflow.set_tag(
                "dataset",
                "European Credit Card Fraud"
            )

            # Log model
            signature = infer_signature(
                X_train.head(5),
                model.predict_proba(X_train.head(5))
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=X_train.head(5),
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

            results[model_name] = {
                "model": model,
                "threshold": threshold,
                "metrics": metrics,
                "run_id": run.info.run_id
            }

            print("Validation metrics:", metrics)

    return results


# ---------------- REGISTER CHAMPION ----------------

def register_champion(best_result):

    model_uri = (
        f"runs:/{best_result['run_id']}/model"
    )

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    client = MlflowClient()

    # Point champion to the newly registered version
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=str(registered_model.version)
    )

    print("\n========== MODEL REGISTRY ==========")

    print("Model:", MODEL_NAME)
    print("Version:", registered_model.version)
    print("Alias: champion")


# ---------------- SAVE LOCAL MODEL ----------------

def save_model(best_result, model_name, feature_names):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "fraud_model.joblib"

    artifact = {
        "model": best_result["model"],
        "threshold": best_result["threshold"],
        "model_name": model_name,
        "feature_names": feature_names
    }

    joblib.dump(artifact, model_path)

    print("\nLocal model saved:", model_path)


# ---------------- FINAL TEST EVALUATION ----------------

def evaluate_test(best_result, X_test, y_test):

    probabilities = best_result["model"].predict_proba(
        X_test
    )[:, 1]

    threshold = best_result["threshold"]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_test,
        probabilities,
        threshold
    )

    print("\n========== FINAL TEST RESULTS ==========")

    print(json.dumps(metrics, indent=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


# ---------------- MAIN ----------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaluate-test",
        action="store_true"
    )

    args = parser.parse_args()

    initialize_mlflow()

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_data()

    results = train_models(
        X_train,
        y_train,
        X_val,
        y_val
    )

    # Select best model using validation PR-AUC
    best_name = max(
        results,
        key=lambda name: results[name]["metrics"]["pr_auc"]
    )

    best_result = results[best_name]

    print("\n========== SELECTED MODEL ==========")

    print("Model:", best_name)

    print("Validation metrics:")
    print(best_result["metrics"])

    # Save local artifact
    save_model(
        best_result,
        best_name,
        list(X_train.columns)
    )

    # Register champion in MLflow
    register_champion(best_result)

    # Test only when explicitly requested
    if args.evaluate_test:

        evaluate_test(
            best_result,
            X_test,
            y_test
        )

    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
