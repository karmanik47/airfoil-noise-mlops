import pytest

from airfoil_noise.modeling.metrics import (
    calculate_regression_metrics,
)


def test_calculates_regression_metrics() -> None:
    metrics = calculate_regression_metrics(
        y_true=[1.0, 2.0, 3.0],
        y_predicted=[1.0, 2.0, 4.0],
    )

    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx((1 / 3) ** 0.5)
    assert metrics["r2"] == pytest.approx(0.5)


def test_perfect_prediction() -> None:
    metrics = calculate_regression_metrics(
        y_true=[1.0, 2.0, 3.0],
        y_predicted=[1.0, 2.0, 3.0],
    )

    assert metrics == {
        "mae": 0.0,
        "rmse": 0.0,
        "r2": 1.0,
    }
