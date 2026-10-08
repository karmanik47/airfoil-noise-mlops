import pandas as pd
import pytest

from airfoil_noise.modeling.train import create_final_datasets


def test_create_final_datasets() -> None:
    dataset = pd.DataFrame(
        {
            "feature": [10.0, 20.0, 30.0, 40.0],
            "target": [1.0, 2.0, 3.0, 4.0],
            "split": [
                "train",
                "validation",
                "test",
                "train",
            ],
        }
    )

    x_training, y_training, x_test, y_test = create_final_datasets(
        dataset=dataset,
        feature_columns=["feature"],
        target_column="target",
    )

    assert x_training["feature"].tolist() == [
        10.0,
        20.0,
        40.0,
    ]
    assert y_training.tolist() == [1.0, 2.0, 4.0]
    assert x_test["feature"].tolist() == [30.0]
    assert y_test.tolist() == [3.0]


def test_create_final_datasets_requires_test_rows() -> None:
    dataset = pd.DataFrame(
        {
            "feature": [10.0, 20.0],
            "target": [1.0, 2.0],
            "split": ["train", "validation"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Тестовая выборка пуста",
    ):
        create_final_datasets(
            dataset=dataset,
            feature_columns=["feature"],
            target_column="target",
        )
