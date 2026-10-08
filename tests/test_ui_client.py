from unittest.mock import Mock, patch

import requests

from airfoil_noise.ui.client import (
    api_is_available,
    request_prediction,
)


def test_api_is_available() -> None:
    response = Mock()

    with patch(
        "airfoil_noise.ui.client.requests.get",
        return_value=response,
    ) as get_request:
        result = api_is_available("http://api.example.com")

    assert result is True
    get_request.assert_called_once_with(
        "http://api.example.com/health",
        timeout=2,
    )
    response.raise_for_status.assert_called_once_with()


def test_api_is_not_available() -> None:
    with patch(
        "airfoil_noise.ui.client.requests.get",
        side_effect=requests.ConnectionError,
    ):
        result = api_is_available("http://api.example.com")

    assert result is False


def test_request_prediction() -> None:
    response = Mock()
    response.json.return_value = {
        "predicted_scaled_sound_pressure": 123.45,
        "unit": "dB",
        "model_name": "random_forest",
        "preprocessing_version": "v1",
    }

    payload = {
        "frequency": 1000.0,
        "angle_of_attack": 5.0,
        "chord_length": 0.15,
        "free_stream_velocity": 55.0,
        "suction_side_displacement_thickness": 0.003,
    }

    with patch(
        "airfoil_noise.ui.client.requests.post",
        return_value=response,
    ) as post_request:
        result = request_prediction(
            api_url="http://api.example.com",
            payload=payload,
        )

    assert result["predicted_scaled_sound_pressure"] == 123.45
    post_request.assert_called_once_with(
        "http://api.example.com/predict",
        json=payload,
        timeout=10,
    )
    response.raise_for_status.assert_called_once_with()
