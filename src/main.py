
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mlflow import MlflowClient


# ---------------- CONFIGURATION ----------------

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "fraud_model.joblib"

MODEL_NAME = "FraudGuard"
MODEL_ALIAS = "champion"

MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

FEATURE_NAMES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]


# ---------------- REQUEST SCHEMA ----------------

class TransactionRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    Time: float = Field(ge=0)

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float = Field(ge=0)


# ---------------- MODEL LOADING ----------------

def load_local_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    return {
        "model": artifact["model"],
        "threshold": float(artifact["threshold"]),
        "model_name": artifact["model_name"],
        "version": "local",
        "feature_names": artifact["feature_names"]
    }


def load_mlflow_model():

    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()

    # Get the currently assigned champion version
    version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    # Load the exact model referenced by the alias
    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    # Retrieve threshold from the same training run
    run = client.get_run(version.run_id)

    threshold = float(
        run.data.params["classification_threshold"]
    )

    return {
        "model": model,
        "threshold": threshold,
        "model_name": MODEL_NAME,
        "version": str(version.version),
        "feature_names": FEATURE_NAMES
    }


# ---------------- APPLICATION LIFESPAN ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading fraud detection model...")

    try:

        if MODEL_SOURCE == "mlflow":
            artifact = load_mlflow_model()

        elif MODEL_SOURCE == "local":
            artifact = load_local_model()

        else:
            raise ValueError(
                "MODEL_SOURCE must be 'local' or 'mlflow'"
            )

        if artifact["feature_names"] != FEATURE_NAMES:
            raise ValueError("Model feature schema mismatch")

        app.state.model = artifact["model"]
        app.state.threshold = artifact["threshold"]
        app.state.model_name = artifact["model_name"]
        app.state.model_version = artifact["version"]

        print("Model loaded successfully!")
        print("Model:", artifact["model_name"])
        print("Version:", artifact["version"])
        print("Threshold:", artifact["threshold"])

    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize fraud model: {exc}"
        ) from exc

    yield


# ---------------- FASTAPI APPLICATION ----------------

app = FastAPI(
    title="FraudGuard API",
    description="Credit card fraud detection service",
    version="1.0.0",
    lifespan=lifespan
)


# ---------------- HEALTH CHECK ----------------

@app.get("/health")
def health_check():

    if not hasattr(app.state, "model"):
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    return {
        "status": "healthy",
        "model": app.state.model_name,
        "version": app.state.model_version
    }


# ---------------- PREDICTION ENDPOINT ----------------

@app.post("/predict")
def predict(transaction: TransactionRequest):

    try:

        # Preserve training feature order
        input_data = pd.DataFrame(
            [[
                getattr(transaction, feature)
                for feature in FEATURE_NAMES
            ]],
            columns=FEATURE_NAMES
        )

        # Obtain fraud probability
        probability = float(
            app.state.model.predict_proba(input_data)[0][1]
        )

        # Apply validation-selected threshold
        prediction = int(
            probability >= app.state.threshold
        )

        return {
            "prediction": prediction,
            "result": (
                "FRAUD" if prediction == 1 else "LEGITIMATE"
            ),
            "fraud_probability": round(probability, 6),
            "threshold": round(app.state.threshold, 6),
            "model": app.state.model_name,
            "model_version": app.state.model_version
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from exc
