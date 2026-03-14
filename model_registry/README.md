# Model Registry

Простой реестр ML-моделей: REST API + SQLite + файловое хранилище.

## Запуск (Docker)

```bash
./run_container.sh
```

Или через docker-compose:
```bash
docker compose up --build
```

API: http://localhost:8000  
Документация: http://localhost:8000/docs

Данные (БД и модели) сохраняются в `./data/`.

## Запуск (локально)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Примеры

**Зарегистрировать модель:**
```bash
curl -X POST "http://localhost:8000/models" \
  -F "name=my_model" \
  -F "team=mlds_1" \
  -F "metadata={\"dataset\": \"train_v1\"}" \
  -F "tags={\"task\": \"classification\"}" \
  -F "status=staging" \
  -F "file=@/path/to/model.pkl"
```

**Список моделей:**
```bash
curl "http://localhost:8000/models?team=mlds_1"
curl "http://localhost:8000/models?status=production"
```

**Скачать модель:**
```bash
curl -O "http://localhost:8000/models/mlds_1_my_model/versions/1"
```

**Обновить статус:**
```bash
curl -X PATCH "http://localhost:8000/models/mlds_1_my_model/versions/1" \
  -H "Content-Type: application/json" \
  -d '{"status": "production"}'
```

## Тестирование

```bash
./run_tests.sh
```

## Переменные окружения

- `REGISTRY_DB` — путь к SQLite (по умолчанию `registry.db`)
- `REGISTRY_STORAGE` — директория для артефактов (по умолчанию `models_storage`)
