from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from airfoil_noise.config import load_config
from airfoil_noise.data.download import calculate_sha256
from airfoil_noise.modeling.metrics import calculate_regression_metrics
from airfoil_noise.modeling.models import build_model_candidates

PREPROCESSING_VERSIONS = ("v1", "v2")
NON_FEATURE_COLUMNS = {"row_id", "split"}


def get_git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_dataset_version(
    processed_data_dir: Path,
    version: str,
) -> tuple[pd.DataFrame, dict[str, object], Path]:
    version_directory = processed_data_dir / version
    dataset_path = version_directory / "dataset.csv"
    manifest_path = version_directory / "manifest.json"

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Подготовленный датасет не найден: {dataset_path}. "
            "Сначала выполните uv run dvc repro."
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Манифест датасета не найден: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(manifest, dict):
        raise TypeError("Манифест должен содержать JSON-объект")

    expected_hash = str(manifest["dataset_sha256"])
    actual_hash = calculate_sha256(dataset_path)

    if actual_hash != expected_hash:
        raise ValueError(
            f"Контрольная сумма {dataset_path} не совпадает "
            "с контрольной суммой в манифесте"
        )

    dataset = pd.read_csv(dataset_path)

    return dataset, manifest, manifest_path


def get_feature_columns(
    dataset: pd.DataFrame,
    target_column: str,
) -> list[str]:
    excluded_columns = NON_FEATURE_COLUMNS | {target_column}

    return [column for column in dataset.columns if column not in excluded_columns]


def select_dataset_split(
    dataset: pd.DataFrame,
    split_name: str,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    selected_rows = dataset.loc[dataset["split"] == split_name]

    if selected_rows.empty:
        raise ValueError(f"Часть датасета {split_name!r} пуста")

    features = selected_rows[feature_columns]
    target = selected_rows[target_column]

    return features, target


def get_model_parameters(
    pipeline: Pipeline,
) -> dict[str, str]:
    estimator = pipeline.named_steps["model"]

    return {
        f"model__{name}": str(value)
        for name, value in estimator.get_params(deep=False).items()
    }


def run_experiments() -> pd.DataFrame:
    config = load_config()
    git_commit = get_git_commit_sha()

    mlflow.set_tracking_uri(config.tracking.uri)
    mlflow.set_experiment(config.tracking.experiment_name)

    results: list[dict[str, object]] = []

    for preprocessing_version in PREPROCESSING_VERSIONS:
        dataset, manifest, manifest_path = load_dataset_version(
            processed_data_dir=config.train.processed_data_dir,
            version=preprocessing_version,
        )

        feature_columns = get_feature_columns(
            dataset=dataset,
            target_column=config.train.target_column,
        )

        x_train, y_train = select_dataset_split(
            dataset=dataset,
            split_name="train",
            feature_columns=feature_columns,
            target_column=config.train.target_column,
        )
        x_validation, y_validation = select_dataset_split(
            dataset=dataset,
            split_name="validation",
            feature_columns=feature_columns,
            target_column=config.train.target_column,
        )

        models = build_model_candidates(random_state=config.project.random_state)

        for model_name, model in models.items():
            run_name = f"{model_name}-{preprocessing_version}"

            with mlflow.start_run(run_name=run_name) as active_run:
                model.fit(x_train, y_train)
                predictions = model.predict(x_validation)

                metrics = calculate_regression_metrics(
                    y_true=y_validation,
                    y_predicted=predictions,
                )

                parameters = {
                    "model_name": model_name,
                    "preprocessing_version": preprocessing_version,
                    "random_state": config.project.random_state,
                    "feature_count": len(feature_columns),
                    "feature_names": ",".join(feature_columns),
                    "train_rows": len(x_train),
                    "validation_rows": len(x_validation),
                    **get_model_parameters(model),
                }

                mlflow.log_params(parameters)
                mlflow.log_metrics(
                    {
                        "validation_mae": metrics["mae"],
                        "validation_rmse": metrics["rmse"],
                        "validation_r2": metrics["r2"],
                    }
                )
                mlflow.set_tags(
                    {
                        "git_commit": git_commit,
                        "dataset_sha256": str(manifest["dataset_sha256"]),
                        "source_sha256": str(manifest["source_sha256"]),
                        "experiment_stage": "model_selection",
                    }
                )

                mlflow.log_artifact(
                    str(manifest_path),
                    artifact_path="data",
                )

                signature = infer_signature(
                    x_train,
                    model.predict(x_train),
                )

                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    signature=signature,
                    input_example=x_train.head(5),
                    serialization_format="cloudpickle",
                )

                results.append(
                    {
                        "run_id": active_run.info.run_id,
                        "model_name": model_name,
                        "preprocessing_version": (preprocessing_version),
                        "feature_count": len(feature_columns),
                        "validation_mae": metrics["mae"],
                        "validation_rmse": metrics["rmse"],
                        "validation_r2": metrics["r2"],
                        "git_commit": git_commit,
                        "model_uri": model_info.model_uri,
                    }
                )

    results_table = pd.DataFrame(results).sort_values(
        by="validation_mae",
        ascending=True,
    )

    reports_directory = Path("reports")
    reports_directory.mkdir(exist_ok=True)

    results_path = reports_directory / "experiment_results.csv"
    results_table.to_csv(
        results_path,
        index=False,
        lineterminator="\n",
    )

    print("\nРезультаты экспериментов:")
    print(results_table.to_string(index=False))
    print(f"\nТаблица сохранена: {results_path}")

    return results_table


def main() -> None:
    run_experiments()


if __name__ == "__main__":
    main()
