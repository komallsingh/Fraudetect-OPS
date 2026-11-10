
import joblib
import numpy as np

from sklearn.dummy import DummyClassifier
from fastapi.testclient import TestClient

from src import main


def test_prediction_api(monkeypatch, tmp_path):

    model_path = tmp_path / "fraud_model.joblib"

    # Create a small dummy classifier
    X = np.zeros((10, len(main.FEATURE_NAMES)))
    y = np.array([0, 1] * 5)

    model = DummyClassifier(strategy="prior")
    model.fit(X, y)

    # Save temporary model artifact
    joblib.dump({
        "model": model,
        "threshold": 0.5,
        "model_name": "test_model",
        "feature_names": main.FEATURE_NAMES
    }, model_path)

    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "MODEL_SOURCE", "local")

    payload = {
        feature: 0.0
        for feature in main.FEATURE_NAMES
    }

    with TestClient(main.app) as client:

        # Health check
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # Prediction
        response = client.post("/predict", json=payload)

        assert response.status_code == 200

        result = response.json()

        assert result["prediction"] in [0, 1]
        assert result["result"] in ["FRAUD", "LEGITIMATE"]
        assert 0 <= result["fraud_probability"] <= 1

        # Missing feature should fail validation
        invalid_payload = payload.copy()
        invalid_payload.pop("V1")

        response = client.post(
            "/predict",
            json=invalid_payload
        )

        assert response.status_code == 422
