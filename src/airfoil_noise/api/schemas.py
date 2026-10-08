from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    frequency: float = Field(gt=0)
    angle_of_attack: float = Field(ge=0)
    chord_length: float = Field(gt=0)
    free_stream_velocity: float = Field(gt=0)
    suction_side_displacement_thickness: float = Field(gt=0)


class PredictionResponse(BaseModel):
    predicted_scaled_sound_pressure: float
    unit: str
    model_name: str
    preprocessing_version: str
