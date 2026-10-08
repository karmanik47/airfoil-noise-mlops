import requests


class ApiClientError(RuntimeError):
    pass


def api_is_available(api_url: str) -> bool:
    try:
        response = requests.get(
            f"{api_url}/health",
            timeout=2,
        )
        response.raise_for_status()
    except requests.RequestException:
        return False

    return True


def request_prediction(
    api_url: str,
    payload: dict[str, float],
) -> dict[str, object]:
    try:
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as error:
        raise ApiClientError("Не удалось получить ответ от API") from error

    if not isinstance(result, dict):
        raise ApiClientError("API вернул ответ в неверном формате")

    return result
