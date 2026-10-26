import json
from pathlib import Path

import pandas as pd
import pytest

from airfoil_noise.api.predictor import ModelPredictor


class RecordingModel:
    def __init__(self) -> None:
        self.received_columns: list[str] = []
        self.received_values: dict[str, float] = {}

    def predict(
        self,
        dataset: pd.DataFrame,
    ) -> list[float]:
        self.received_columns = list(dataset.columns)
        self.received_values = {
            str(name): float(value) for name, value in dataset.iloc[0].items()
        }

        return [123.45]


def test_predictor_uses_metadata_feature_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"

    model_path.write_bytes(b"test-model")
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "test-model",
                "preprocessing_version": "v1",
                "model_sha256": "test-sha256",
                "feature_columns": [
                    "first_feature",
                    "second_feature",
                ],
            }
        ),
        encoding="utf-8",
    )

    recording_model = RecordingModel()

    def fake_load(path: Path) -> RecordingModel:
        assert path == model_path
        return recording_model

    monkeypatch.setattr(
        "airfoil_noise.api.predictor.joblib.load",
        fake_load,
    )

    predictor = ModelPredictor(
        model_path=model_path,
        metadata_path=metadata_path,
    )

    prediction = predictor.predict(
        {
            "second_feature": 2.0,
            "first_feature": 1.0,
        }
    )

    assert prediction == 123.45
    assert recording_model.received_columns == [
        "first_feature",
        "second_feature",
    ]
    assert recording_model.received_values == {
        "first_feature": 1.0,
        "second_feature": 2.0,
    }
