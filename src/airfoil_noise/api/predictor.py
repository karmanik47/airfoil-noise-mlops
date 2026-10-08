import json
from collections.abc import Mapping
from pathlib import Path

import joblib
import pandas as pd


class ModelPredictor:
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Файл модели не найден: {model_path}. "
                "Выполните uv run train-final-model."
            )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Файл метаданных не найден: {metadata_path}. "
                "Выполните uv run train-final-model."
            )

        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if not isinstance(raw_metadata, dict):
            raise TypeError("Метаданные модели должны содержать JSON-объект")

        feature_columns = raw_metadata.get("feature_columns")

        if not isinstance(feature_columns, list) or not all(
            isinstance(column, str) for column in feature_columns
        ):
            raise TypeError("В метаданных отсутствует список feature_columns")

        self.model = joblib.load(model_path)
        self.feature_columns = feature_columns
        self.model_name = str(raw_metadata["model_name"])
        self.preprocessing_version = str(raw_metadata["preprocessing_version"])

    def predict(
        self,
        features: Mapping[str, float],
    ) -> float:
        missing_features = [
            column for column in self.feature_columns if column not in features
        ]

        if missing_features:
            raise ValueError("Не переданы признаки: " + ", ".join(missing_features))

        model_input = pd.DataFrame(
            [{column: features[column] for column in self.feature_columns}]
        )

        prediction = self.model.predict(model_input)

        return float(prediction[0])
