from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = os.getenv(
    "AIRFOIL_API_URL",
    "http://localhost:8000",
).rstrip("/")


def send_request(
    path: str,
    method: str = "GET",
    payload: dict[str, float] | None = None,
) -> tuple[int, bytes]:
    body = None
    headers: dict[str, str] = {}

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        url=f"{BASE_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def send_json_request(
    path: str,
    method: str = "GET",
    payload: dict[str, float] | None = None,
) -> tuple[int, dict[str, object]]:
    status, body = send_request(
        path=path,
        method=method,
        payload=payload,
    )

    result = json.loads(body.decode("utf-8"))

    if not isinstance(result, dict):
        raise TypeError(f"{path} вернул ответ в неверном формате")

    return status, result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wait_for_health(
    timeout_seconds: float = 60,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            status, result = send_json_request("/health")

            if status == 200:
                return result
        except (URLError, ValueError, TypeError) as error:
            last_error = error

        time.sleep(1)

    raise RuntimeError("API не стал доступен за отведённое время") from last_error


def main() -> None:
    print(f"Проверка API: {BASE_URL}", flush=True)
    health = wait_for_health()

    require(
        health.get("status") == "ok",
        "/health не подтвердил готовность сервиса",
    )
    require(
        health.get("model_name") == "random_forest",
        "/health вернул неверную модель",
    )
    require(
        health.get("preprocessing_version") == "v1",
        "/health вернул неверную версию preprocessing",
    )

    model_sha256 = health.get("model_sha256")

    require(
        isinstance(model_sha256, str) and len(model_sha256) == 64,
        "/health не вернул корректный SHA256 модели",
    )

    print("[OK] /health и загрузка модели")

    valid_payload = {
        "frequency": 1000.0,
        "angle_of_attack": 5.0,
        "chord_length": 0.15,
        "free_stream_velocity": 55.0,
        "suction_side_displacement_thickness": 0.003,
    }

    status, prediction = send_json_request(
        path="/predict",
        method="POST",
        payload=valid_payload,
    )

    require(
        status == 200,
        "Корректный /predict не вернул HTTP 200",
    )
    require(
        isinstance(
            prediction.get("predicted_scaled_sound_pressure"),
            int | float,
        ),
        "/predict не вернул числовой прогноз",
    )
    require(
        prediction.get("model_sha256") == model_sha256,
        "Версия модели в /predict не совпадает с /health",
    )

    print("[OK] корректный /predict")

    invalid_payload = {
        **valid_payload,
        "frequency": -1000.0,
    }

    status, error_response = send_json_request(
        path="/predict",
        method="POST",
        payload=invalid_payload,
    )

    require(
        status == 422,
        "Некорректный /predict не вернул HTTP 422",
    )
    require(
        "detail" in error_response,
        "Ошибка валидации не содержит detail",
    )

    print("[OK] некорректный /predict возвращает 422")

    docs_status, _ = send_request("/docs")

    require(
        docs_status == 200,
        "/docs не вернул HTTP 200",
    )

    print("[OK] /docs")
    print("Smoke-тест успешно пройден")


if __name__ == "__main__":
    main()
