# Оптимизация inference pipeline: rubert-mini-frida

Три варианта HTTP-сервиса для расчёта эмбеддингов — базовый (HuggingFace), ONNX, ONNX с динамическим батчированием.

Команды для запуска и выполнения ДЗ — см. **RUN_COMMANDS.md**.

## API

Все три сервиса реализуют одинаковый контракт:

```
POST /embed
{"texts": ["текст 1", "текст 2"], "prompt": "categorize: "}

→ {"embeddings": [[...], [...]]}
```

Swagger: `http://localhost:{port}/docs`

## Структура

```
config.py          — константы (модель, префикс, max_length)
export_onnx.py     — конвертация в ONNX через optimum
part1_basic/       — SentenceTransformer + FastAPI
part2_onnx/        — onnx-runtime + FastAPI
part3_batching/    — ONNX + asyncio queue/worker (динамический батч)
REPORT.md          — описание подхода и результаты бенчмарков
```

## 1. Сборка образов

```bash
docker compose build
```

---

## 2. Экспорт ONNX (один раз, нужен для частей 2 и 3)

```bash
docker compose run --rm export
```

---

## 3. Часть 1 — базовый инференс


```bash
docker compose up part1
```

```bash
docker compose run --rm benchmark1
```

Результаты (latency, throughput) — в выводе. (Потребление ресурсов - отдельно через docker stats)

---

## 4. Часть 2 — ONNX

```bash
docker compose up part2
```

```bash
docker compose run --rm benchmark2
```

---

## 5. Часть 3 — батчирование

```bash
docker compose up part3
```

```bash
docker compose run --rm benchmark3
```


Результаты — в REPORT.md. 
