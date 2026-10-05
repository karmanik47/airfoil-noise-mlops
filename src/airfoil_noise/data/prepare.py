from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from airfoil_noise.config import load_config
from airfoil_noise.data.download import RAW_SHA256, calculate_sha256

FEATURE_COLUMNS = [
    "frequency",
    "angle_of_attack",
    "chord_length",
    "free_stream_velocity",
    "suction_side_displacement_thickness",
]
TARGET_COLUMN = "scaled_sound_pressure"
RAW_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]
EXPECTED_ROWS = 1503


def load_raw_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.is_file():
        message = (
            f"Исходный датасет не найден: {file_path}. "
            "Сначала выполните команду uv run download-data."
        )
        raise FileNotFoundError(message)

    dataset = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
    )

    validate_raw_dataset(dataset)
    return dataset


def validate_raw_dataset(dataset: pd.DataFrame) -> None:
    expected_shape = (EXPECTED_ROWS, len(RAW_COLUMNS))

    if dataset.shape != expected_shape:
        message = (
            f"Неверный размер датасета: {dataset.shape}. Ожидался: {expected_shape}."
        )
        raise ValueError(message)

    if dataset.isna().any().any():
        raise ValueError("В исходном датасете обнаружены пропущенные значения")

    if dataset.duplicated().any():
        raise ValueError("В исходном датасете обнаружены повторяющиеся строки")


def create_split_labels(
    dataset: pd.DataFrame,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> pd.Series:
    holdout_size = validation_size + test_size

    train_indices, holdout_indices = train_test_split(
        dataset.index.to_numpy(),
        test_size=holdout_size,
        random_state=random_state,
    )

    relative_test_size = test_size / holdout_size

    validation_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=relative_test_size,
        random_state=random_state,
    )

    labels = pd.Series(
        index=dataset.index,
        dtype="string",
    )
    labels.loc[train_indices] = "train"
    labels.loc[validation_indices] = "validation"
    labels.loc[test_indices] = "test"

    if labels.isna().any():
        raise RuntimeError("Не всем строкам назначена часть выборки")

    return labels


def create_preprocessing_v1(
    raw_dataset: pd.DataFrame,
    split_labels: pd.Series,
) -> pd.DataFrame:
    dataset = raw_dataset.copy()
    dataset.insert(0, "row_id", range(len(dataset)))
    dataset["split"] = split_labels

    return dataset


def add_engineered_features(dataset: pd.DataFrame) -> pd.DataFrame:
    result = dataset.copy()
    result["log_frequency"] = result["frequency"].map(math.log1p)
    result["log_suction_side_displacement_thickness"] = result[
        "suction_side_displacement_thickness"
    ].map(math.log1p)

    return result


def create_preprocessing_v2(
    preprocessing_v1: pd.DataFrame,
) -> pd.DataFrame:
    dataset = add_engineered_features(preprocessing_v1)

    ordered_columns = [
        "row_id",
        *FEATURE_COLUMNS,
        "log_frequency",
        "log_suction_side_displacement_thickness",
        TARGET_COLUMN,
        "split",
    ]

    return dataset[ordered_columns]


def save_version(
    dataset: pd.DataFrame,
    output_root: Path,
    version: str,
) -> Path:
    version_directory = output_root / version
    version_directory.mkdir(parents=True, exist_ok=True)

    dataset_path = version_directory / "dataset.csv"
    dataset.to_csv(dataset_path, index=False, lineterminator="\n")

    split_counts = {
        str(name): int(count) for name, count in dataset["split"].value_counts().items()
    }

    manifest = {
        "version": version,
        "source_sha256": RAW_SHA256,
        "dataset_sha256": calculate_sha256(dataset_path),
        "rows": len(dataset),
        "columns": list(dataset.columns),
        "split_counts": split_counts,
    }

    manifest_path = version_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return dataset_path


def prepare_datasets() -> tuple[Path, Path]:
    config = load_config()

    raw_dataset = load_raw_dataset(config.train.raw_data_path)
    split_labels = create_split_labels(
        raw_dataset,
        validation_size=config.train.validation_size,
        test_size=config.train.test_size,
        random_state=config.project.random_state,
    )

    preprocessing_v1 = create_preprocessing_v1(
        raw_dataset,
        split_labels,
    )
    preprocessing_v2 = create_preprocessing_v2(preprocessing_v1)

    v1_path = save_version(
        preprocessing_v1,
        config.train.processed_data_dir,
        "v1",
    )
    v2_path = save_version(
        preprocessing_v2,
        config.train.processed_data_dir,
        "v2",
    )

    return v1_path, v2_path


def main() -> None:
    v1_path, v2_path = prepare_datasets()
    print(f"Создана версия preprocessing v1: {v1_path}")
    print(f"Создана версия preprocessing v2: {v2_path}")


if __name__ == "__main__":
    main()
