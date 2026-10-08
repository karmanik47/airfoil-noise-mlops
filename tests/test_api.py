from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from airfoil_noise.api.app import app, get_predictor


class FakePredictor:
    model_name = "test-model"
    preprocessing_version = "v1"

    def predict(
        self,
        features: dict[str, float],
    ) -> float:
        return 123.45


@pytest.fixture
def client() -> Generator[TestClient]:
    app.dependency_overrides[get_predictor] = FakePredictor

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "airfoil-noise-api",
    }


def test_predict(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "frequency": 1000.0,
            "angle_of_attack": 5.0,
            "chord_length": 0.15,
            "free_stream_velocity": 55.0,
            "suction_side_displacement_thickness": 0.003,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "predicted_scaled_sound_pressure": 123.45,
        "unit": "dB",
        "model_name": "test-model",
        "preprocessing_version": "v1",
    }


def test_predict_rejects_invalid_values(
    client: TestClient,
) -> None:
    response = client.post(
        "/predict",
        json={
            "frequency": -1000.0,
            "angle_of_attack": 5.0,
            "chord_length": 0.15,
            "free_stream_velocity": 55.0,
            "suction_side_displacement_thickness": 0.003,
        },
    )

    assert response.status_code == 422
