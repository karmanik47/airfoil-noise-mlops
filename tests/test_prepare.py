import math

import pandas as pd
import pytest

from airfoil_noise.data.prepare import (
    RAW_COLUMNS,
    add_engineered_features,
    create_split_labels,
    validate_raw_dataset,
)


def test_split_is_reproducible() -> None:
    dataset = pd.DataFrame(index=range(100))

    first_split = create_split_labels(
        dataset,
        validation_size=0.15,
        test_size=0.15,
        random_state=42,
    )
    second_split = create_split_labels(
        dataset,
        validation_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    pd.testing.assert_series_equal(first_split, second_split)

    assert first_split.value_counts().to_dict() == {
        "train": 70,
        "test": 15,
        "validation": 15,
    }


def test_engineered_features() -> None:
    dataset = pd.DataFrame(
        {
            "frequency": [800.0],
            "suction_side_displacement_thickness": [0.00266337],
        }
    )

    result = add_engineered_features(dataset)

    assert result.loc[0, "log_frequency"] == pytest.approx(math.log1p(800.0))
    assert result.loc[
        0,
        "log_suction_side_displacement_thickness",
    ] == pytest.approx(math.log1p(0.00266337))


def test_invalid_dataset_shape() -> None:
    dataset = pd.DataFrame(
        [[1, 2, 3, 4, 5, 6]],
        columns=RAW_COLUMNS,
    )

    with pytest.raises(ValueError, match="Неверный размер датасета"):
        validate_raw_dataset(dataset)
