from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from airfoil_noise.modeling.models import build_model_candidates


def test_builds_three_model_candidates() -> None:
    models = build_model_candidates(random_state=42)

    assert set(models) == {
        "ridge",
        "random_forest",
        "gradient_boosting",
    }


def test_ridge_uses_standard_scaling() -> None:
    ridge_pipeline = build_model_candidates(42)["ridge"]

    assert isinstance(
        ridge_pipeline.named_steps["scaler"],
        StandardScaler,
    )
    assert isinstance(
        ridge_pipeline.named_steps["model"],
        Ridge,
    )
    assert ridge_pipeline.named_steps["model"].alpha == 1.0


def test_tree_models_are_reproducible() -> None:
    models = build_model_candidates(random_state=42)

    random_forest = models["random_forest"].named_steps["model"]
    gradient_boosting = models["gradient_boosting"].named_steps["model"]

    assert isinstance(random_forest, RandomForestRegressor)
    assert random_forest.random_state == 42
    assert random_forest.n_estimators == 300
    assert random_forest.min_samples_leaf == 2

    assert isinstance(
        gradient_boosting,
        GradientBoostingRegressor,
    )
    assert gradient_boosting.random_state == 42
    assert gradient_boosting.n_estimators == 200
    assert gradient_boosting.learning_rate == 0.05
    assert gradient_boosting.max_depth == 3
