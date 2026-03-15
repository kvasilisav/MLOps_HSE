import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from config import DEFAULT_PROMPT, MAX_LENGTH

ONNX_DIR = Path(__file__).resolve().parents[1] / "onnx_model"

BATCH_WAIT_SEC = 0.05
BATCH_MAX_SIZE = 32

_session = None
_tokenizer = None
_queue: asyncio.Queue = None


def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)
    s = (last_hidden_state * mask[:, :, np.newaxis]).sum(axis=1)
    d = mask.sum(axis=1, keepdims=True)
    return s / np.where(d == 0, 1, d)


def _normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.where(norm == 0, 1, norm)).astype(np.float32)


def get_session():
    global _session, _tokenizer
    if _session is None:
        import onnxruntime as ort
        onnx_path = ONNX_DIR / "model.onnx"
        if not onnx_path.exists():
            onnx_path = next(ONNX_DIR.glob("*.onnx"), None)
        if onnx_path is None:
            raise FileNotFoundError(f"No ONNX file in {ONNX_DIR}. Run: python export_onnx.py")
        _session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        _tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
    return _session, _tokenizer


async def _worker():
    while True:
        batch = []
        try:
            batch.append(await asyncio.wait_for(_queue.get(), timeout=BATCH_WAIT_SEC))
            while len(batch) < BATCH_MAX_SIZE:
                try:
                    batch.append(_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
        except asyncio.TimeoutError:
            continue

        all_texts = [f"{item['prompt']}{t}" for item in batch for t in item["texts"]]
        session, tokenizer = get_session()
        enc = tokenizer(all_texts, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="np")
        if "token_type_ids" not in enc:
            enc["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)
        inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "token_type_ids": enc["token_type_ids"].astype(np.int64),
        }
        out = await asyncio.to_thread(session.run, None, inputs)
        emb_list = _normalize(_mean_pool(out[0], enc["attention_mask"])).tolist()

        offset = 0
        for item in batch:
            n = len(item["texts"])
            item["future"].set_result(emb_list[offset: offset + n])
            offset += n


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _queue
    get_session()
    _queue = asyncio.Queue()
    asyncio.create_task(_worker())
    yield


app = FastAPI(title="Embedding API (dynamic batching)", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str]
    prompt: str = DEFAULT_PROMPT


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    future = asyncio.get_running_loop().create_future()
    await _queue.put({"texts": req.texts, "prompt": req.prompt, "future": future})
    return EmbedResponse(embeddings=await future)
