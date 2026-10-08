from functools import lru_cache
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI

from airfoil_noise.api.predictor import ModelPredictor
from airfoil_noise.api.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from airfoil_noise.config import load_config

app = FastAPI(
    title="Airfoil Noise API",
    description=("API для прогнозирования уровня аэродинамического шума"),
    version="0.1.0",
)


@lru_cache
def get_predictor() -> ModelPredictor:
    config = load_config()

    return ModelPredictor(
        model_path=config.inference.model_path,
        metadata_path=config.inference.metadata_path,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "airfoil-noise-api",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
    predictor: Annotated[
        ModelPredictor,
        Depends(get_predictor),
    ],
) -> PredictionResponse:
    prediction = predictor.predict(request.model_dump())

    return PredictionResponse(
        predicted_scaled_sound_pressure=prediction,
        unit="dB",
        model_name=predictor.model_name,
        preprocessing_version=(predictor.preprocessing_version),
    )


def main() -> None:
    config = load_config()

    uvicorn.run(
        app,
        host=config.inference.host,
        port=config.inference.port,
    )


if __name__ == "__main__":
    main()
