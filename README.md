## Структура репозитория

| Путь | Назначение |
|---|---|
| `data/` | Исходные данные. Содержит `spotify_data.csv|
| `task_1/` | Задача 1 |
| `task_2/` | Задача 2 |
| `task_3/` | Задача 3 |
| `task_3/notebooks/` | Ноутбуки и Python-модули задачи 3 |
| `task_3/data/splits/` | Зафиксированные train/val/test parquet-выборки |
| `task_3/models/` | Обученные модели: одиночный CatBoost (`catboost_genre.cbm`) и ансамбль (`ensemble/`). |
| `task_4/` | Задача 4 |
| `pyproject.toml`, `uv.lock`, `.python-version` | Конфигурация окружения. Управление через **uv**, Python **3.12**. |

## Быстрый старт

```bash
uv sync                                   # установить зависимости в .venv
uv run jupyter lab                        # открыть ноутбуки
uv run python task_3/notebooks/_build.py  # пересобрать ноутбуки задачи 3 из _build.py
```
