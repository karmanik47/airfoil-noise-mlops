import os
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator


class ProjectConfig(BaseModel):
    name: str = Field(min_length=1)
    random_state: int = Field(ge=0)


class TrainConfig(BaseModel):
    raw_data_path: Path
    processed_data_dir: Path
    target_column: str = Field(min_length=1)
    validation_size: float = Field(gt=0, lt=1)
    test_size: float = Field(gt=0, lt=1)
    preprocessing_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_split_sizes(self) -> Self:
        if self.validation_size + self.test_size >= 1:
            raise ValueError("Сумма validation_size и test_size должна быть меньше 1")
        return self


class InferenceConfig(BaseModel):
    model_path: Path
    metadata_path: Path
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


class TrackingConfig(BaseModel):
    uri: str = Field(min_length=1)
    experiment_name: str = Field(min_length=1)


class AppConfig(BaseModel):
    project: ProjectConfig
    train: TrainConfig
    inference: InferenceConfig
    tracking: TrackingConfig


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = Path(config_path or os.getenv("AIRFOIL_CONFIG_PATH", "configs/config.yaml"))

    if not path.is_file():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

    with path.open(encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise TypeError("Конфигурация должна содержать YAML-объект")

    inference = raw_config.setdefault("inference", {})
    inference_overrides = {
        "AIRFOIL_MODEL_PATH": "model_path",
        "AIRFOIL_HOST": "host",
        "AIRFOIL_PORT": "port",
    }

    for environment_name, config_name in inference_overrides.items():
        value = os.getenv(environment_name)
        if value is not None:
            inference[config_name] = value

    tracking = raw_config.setdefault("tracking", {})
    tracking_overrides = {
        "MLFLOW_TRACKING_URI": "uri",
        "MLFLOW_EXPERIMENT_NAME": "experiment_name",
    }

    for environment_name, config_name in tracking_overrides.items():
        value = os.getenv(environment_name)
        if value is not None:
            tracking[config_name] = value

    return AppConfig.model_validate(raw_config)
