from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from airfoil_noise.config import load_config
from airfoil_noise.data.download import calculate_sha256
from airfoil_noise.modeling.experiment import (
    get_feature_columns,
    get_git_commit_sha,
    get_model_parameters,
    load_dataset_version,
)
from airfoil_noise.modeling.metrics import calculate_regression_metrics
from airfoil_noise.modeling.models import build_model_candidates

FINAL_MODEL_NAME = "random_forest"
FINAL_PREPROCESSING_VERSION = "v1"


def create_final_datasets(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    training_rows = dataset["split"].isin(["train", "validation"])
    test_rows = dataset["split"] == "test"

    if not training_rows.any():
        raise ValueError("Обучающая выборка пуста")

    if not test_rows.any():
        raise ValueError("Тестовая выборка пуста")

    x_training = dataset.loc[training_rows, feature_columns]
    y_training = dataset.loc[training_rows, target_column]

    x_test = dataset.loc[test_rows, feature_columns]
    y_test = dataset.loc[test_rows, target_column]

    return x_training, y_training, x_test, y_test


def train_final_model() -> dict[str, object]:
    config = load_config()
    git_commit = get_git_commit_sha()

    dataset, manifest, manifest_path = load_dataset_version(
        processed_data_dir=config.train.processed_data_dir,
        version=FINAL_PREPROCESSING_VERSION,
    )

    feature_columns = get_feature_columns(
        dataset=dataset,
        target_column=config.train.target_column,
    )

    x_training, y_training, x_test, y_test = create_final_datasets(
        dataset=dataset,
        feature_columns=feature_columns,
        target_column=config.train.target_column,
    )

    model = build_model_candidates(random_state=config.project.random_state)[
        FINAL_MODEL_NAME
    ]

    mlflow.set_tracking_uri(config.tracking.uri)
    mlflow.set_experiment(config.tracking.experiment_name)

    with mlflow.start_run(
        run_name=f"final-{FINAL_MODEL_NAME}-{FINAL_PREPROCESSING_VERSION}"
    ) as active_run:
        model.fit(x_training, y_training)
        test_predictions = model.predict(x_test)

        test_metrics = calculate_regression_metrics(
            y_true=y_test,
            y_predicted=test_predictions,
        )

        config.inference.model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        config.inference.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(model, config.inference.model_path)
        model_sha256 = calculate_sha256(config.inference.model_path)

        mlflow.log_params(
            {
                "model_name": FINAL_MODEL_NAME,
                "preprocessing_version": (FINAL_PREPROCESSING_VERSION),
                "random_state": config.project.random_state,
                "feature_count": len(feature_columns),
                "feature_names": ",".join(feature_columns),
                "training_rows": len(x_training),
                "test_rows": len(x_test),
                **get_model_parameters(model),
            }
        )
        mlflow.log_metrics(
            {
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )
        mlflow.set_tags(
            {
                "git_commit": git_commit,
                "dataset_sha256": str(manifest["dataset_sha256"]),
                "source_sha256": str(manifest["source_sha256"]),
                "experiment_stage": "final_evaluation",
                "selection_metric": "validation_mae",
            }
        )

        signature = infer_signature(
            x_training,
            model.predict(x_training),
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=x_training.head(5),
            serialization_format="cloudpickle",
        )

        metadata: dict[str, object] = {
            "model_name": FINAL_MODEL_NAME,
            "preprocessing_version": (FINAL_PREPROCESSING_VERSION),
            "feature_columns": feature_columns,
            "target_column": config.train.target_column,
            "training_rows": len(x_training),
            "test_rows": len(x_test),
            "random_state": config.project.random_state,
            "test_metrics": test_metrics,
            "dataset_sha256": str(manifest["dataset_sha256"]),
            "source_sha256": str(manifest["source_sha256"]),
            "model_sha256": model_sha256,
            "git_commit": git_commit,
            "mlflow_run_id": active_run.info.run_id,
            "mlflow_model_uri": model_info.model_uri,
        }

        metadata_text = (
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )

        config.inference.metadata_path.write_text(
            metadata_text,
            encoding="utf-8",
        )

        reports_directory = Path("reports")
        reports_directory.mkdir(exist_ok=True)

        report_path = reports_directory / "final_test_metrics.json"
        report_path.write_text(
            metadata_text,
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(manifest_path),
            artifact_path="data",
        )
        mlflow.log_artifact(
            str(config.inference.model_path),
            artifact_path="exported_model",
        )
        mlflow.log_artifact(
            str(config.inference.metadata_path),
            artifact_path="exported_model",
        )

    print("\nФинальная модель обучена")
    print(f"Модель: {FINAL_MODEL_NAME}")
    print(f"Preprocessing: {FINAL_PREPROCESSING_VERSION}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"Модель сохранена: {config.inference.model_path}")
    print(f"Метаданные сохранены: {config.inference.metadata_path}")

    return metadata


def main() -> None:
    train_final_model()


if __name__ == "__main__":
    main()
