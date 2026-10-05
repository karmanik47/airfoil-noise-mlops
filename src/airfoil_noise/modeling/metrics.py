from collections.abc import Sequence

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def calculate_regression_metrics(
    y_true: Sequence[float],
    y_predicted: Sequence[float],
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_predicted)),
        "rmse": float(root_mean_squared_error(y_true, y_predicted)),
        "r2": float(r2_score(y_true, y_predicted)),
    }
