from pathlib import Path

import pytest

from airfoil_noise.config import load_config


def test_load_config() -> None:
    config = load_config()

    assert config.project.name == "airfoil-noise-mlops"
    assert config.project.random_state == 42
    assert config.train.target_column == "scaled_sound_pressure"
    assert config.inference.port == 8000


def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRFOIL_PORT", "9000")
    monkeypatch.setenv(
        "AIRFOIL_MODEL_PATH",
        "models/test-model.joblib",
    )

    config = load_config()

    assert config.inference.port == 9000
    assert config.inference.model_path == Path("models/test-model.joblib")


def test_missing_config_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Файл конфигурации не найден",
    ):
        load_config(missing_file)
