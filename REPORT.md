# Итоговый отчёт

## Проект

- Название: Airfoil Noise MLOps
- Репозиторий: https://github.com/karmanik47/airfoil-noise-mlops
- Итоговая версия: `v1.0.0`
- Python: 3.12
- Тип задачи: регрессия
- Дополнительный компонент: Streamlit

Полный commit SHA итоговой версии определяется командой:

```powershell
git rev-list -n 1 v1.0.0
```

Также SHA будет указан в описании GitHub Release `v1.0.0`.

## Результат

Разработан воспроизводимый MLOps-проект, включающий:

- загрузку и проверку датасета;
- две версии preprocessing;
- фиксированное разбиение train, validation и test;
- шесть сравнимых MLflow-экспериментов;
- выбор и обучение итоговой модели;
- версионирование данных с DVC;
- FastAPI-сервис;
- автоматические тесты;
- smoke-тест;
- Streamlit-интерфейс;
- запуск через Docker Compose.

Итоговая модель — Random Forest с preprocessing `v1`.

| Метрика | Test |
|---|---:|
| MAE | 1.4446 |
| RMSE | 1.9284 |
| R² | 0.9165 |

## Идентификаторы

### Репозиторий и версия

- Репозиторий: https://github.com/karmanik47/airfoil-noise-mlops
- Tag: `v1.0.0`
- Release: https://github.com/karmanik47/airfoil-noise-mlops/releases/tag/v1.0.0

### Эксперименты

- MLflow experiment: `airfoil-noise-comparison`
- Итоговый MLflow run ID: `82a0ed10784d48309acddcaf54612b6c`
- Итоговый MLflow model URI: `models:/m-251b70aee7d0430a851e797c385d329c`
- Таблица экспериментов:  
  https://github.com/karmanik47/airfoil-noise-mlops/blob/v1.0.0/reports/experiment_results.csv
- Итоговые метрики:  
  https://github.com/karmanik47/airfoil-noise-mlops/blob/v1.0.0/reports/final_test_metrics.json

### Версии данных

Исходный файл:

```text
SHA256:
74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3
```

Preprocessing `v1`:

```text
SHA256:
c4539e0edffd13ea296e0d657ffa9845a839a5d4fee77da1472160b8c9b7e3c0
```

Preprocessing `v2`:

```text
SHA256:
b966fc4a6ac537b2d9eb1ea18ccb494ccfb3b0d65d33b7f2509c6203a06957b4
```

DVC-файлы:

- https://github.com/karmanik47/airfoil-noise-mlops/blob/v1.0.0/data/raw/airfoil_self_noise.dat.dvc
- https://github.com/karmanik47/airfoil-noise-mlops/blob/v1.0.0/dvc.yaml
- https://github.com/karmanik47/airfoil-noise-mlops/blob/v1.0.0/dvc.lock

### Model artifact

- Файл модели:  
  https://github.com/karmanik47/airfoil-noise-mlops/releases/download/v1.0.0/model.joblib
- Метаданные:  
  https://github.com/karmanik47/airfoil-noise-mlops/releases/download/v1.0.0/metadata.json
- SHA256:

```text
38a3abf4f32306bf67defc660066f5f296e5f0423bcb2d3e17907e60fadbccff
```

## Проверка требований

| Требование | Реализация |
|---|---|
| Отдельный Git-репозиторий | GitHub-репозиторий проекта |
| Основная ветка `main` | `main` |
| Работа через отдельные ветки и PR | История Pull Request в GitHub |
| Python и lock-файл | `pyproject.toml`, `.python-version`, `uv.lock` |
| Игнорирование локальных файлов | `.gitignore`, `.dockerignore` |
| Шаблон окружения | `.env.example` |
| Версионирование данных | DVC, `dvc.yaml`, `dvc.lock` |
| Две версии preprocessing | `data/processed/v1`, `data/processed/v2` |
| Не менее трёх экспериментов | 6 MLflow-запусков |
| Связь с Git и данными | Git SHA и SHA256 записаны в MLflow |
| Отдельная test-выборка | 226 строк, однократная оценка |
| Сохранённая модель | GitHub Release и MLflow model artifact |
| `GET /health` | FastAPI |
| `POST /predict` | FastAPI |
| `/docs` | Swagger, генерируемый FastAPI |
| Схемы запроса и ответа | Pydantic |
| Позитивные и негативные тесты | `tests/` |
| Тест преобразования входа | `tests/test_predictor.py` |
| Конфигурация train/inference | `configs/config.yaml` |
| Переменные окружения | `.env.example`, `config.py`, Streamlit |
| Dockerfile | Зафиксированный base image |
| Запуск не от root | `USER appuser` |
| Версионный image | `airfoil-noise-app:1.0.0` |
| Docker Compose | API и Streamlit |
| Автоматический smoke-тест | `scripts/smoke_test.py` |
| Пользовательский интерфейс | Streamlit обращается к FastAPI |
| README | `README.md` |
| Model Card | `MODEL_CARD.md` |
| Dataset Card | `DATASET_CARD.md` |
| Архитектурная схема | Mermaid в `README.md` |

## Полная проверка на чистом окружении

### 1. Клонирование

```powershell
git clone --branch v1.0.0 https://github.com/karmanik47/airfoil-noise-mlops.git
cd airfoil-noise-mlops
```

### 2. Восстановление окружения

```powershell
uv sync
```

### 3. Получение и подготовка данных

```powershell
uv run download-data
uv run dvc repro
uv run dvc status
```

Ожидаемый результат:

```text
Data and pipelines are up to date.
```

### 4. Воспроизведение экспериментов

```powershell
uv run run-experiments
```

### 5. Получение финальной модели

Можно скачать `model.joblib` и `metadata.json` из GitHub Release `v1.0.0` и поместить в каталог `models`.

Альтернативный способ — повторное обучение:

```powershell
uv run train-final-model
```

### 6. Проверки кода

```powershell
uv run pytest
uv run ruff check .
```

### 7. Docker

```powershell
docker compose build --no-cache
docker compose up -d
docker compose ps
```

Оба контейнера должны получить статус `healthy`.

### 8. Smoke-тест

```powershell
uv run python scripts/smoke_test.py
```

Ожидаемый результат:

```text
[OK] /health и загрузка модели
[OK] корректный /predict
[OK] некорректный /predict возвращает 422
[OK] /docs
Smoke-тест успешно пройден
```

### 9. Остановка

```powershell
docker compose down
```

## Демонстрационное видео

Итоговое видео:

https://github.com/karmanik47/airfoil-noise-mlops/releases/download/v1.0.0/demo.mp4

План видео продолжительностью 1–3 минуты:

1. Показать репозиторий и tag `v1.0.0`.
2. Выполнить `docker compose up -d`.
3. Показать `docker compose ps` со статусом `healthy`.
4. Открыть `/health`.
5. Выполнить корректный `/predict`.
6. Отправить некорректный запрос и показать HTTP 422.
7. Открыть Streamlit и получить прогноз.
8. Выполнить `docker compose down`.

## Проверка секретов и локальных файлов

В Git не добавляются:

- `.env`;
- виртуальное окружение `.venv`;
- локальная база `mlflow.db`;
- каталоги `mlruns` и `mlartifacts`;
- локальные данные;
- локальные model artifacts;
- токены, ключи и пароли.

Docker build context ограничен файлом `.dockerignore`.

## Известные ограничения

- MLflow используется с локальным SQLite backend.
- Интерактивный MLflow UI не публикуется в интернете.
- Итоговые параметры, метрики, run ID и Git SHA экспортированы в версионируемые отчёты.
- Модель предназначена для учебного проекта и не должна применяться в критичных для безопасности расчётах.
- Качество за пределами диапазонов исходного датасета не гарантируется.
- В проекте нет CI/CD; дополнительным компонентом выбран Streamlit.