from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from config import DEFAULT_PROMPT, MAX_LENGTH

ONNX_DIR = Path(__file__).resolve().parents[1] / "onnx_model"

_session = None
_tokenizer = None


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
        onnx_path = ONNX_DIR / "model.onnx"
        if not onnx_path.exists():
            onnx_path = next(ONNX_DIR.glob("*.onnx"), None)
        if onnx_path is None:
            raise FileNotFoundError(f"No ONNX file in {ONNX_DIR}. Run: python export_onnx.py")
        _session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        _tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
    return _session, _tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_session()
    yield


app = FastAPI(title="Embedding API (ONNX)", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str]
    prompt: str = DEFAULT_PROMPT


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    session, tokenizer = get_session()
    prefixed = [f"{req.prompt}{t}" for t in req.texts]
    enc = tokenizer(prefixed, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="np")
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)
    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
        "token_type_ids": enc["token_type_ids"].astype(np.int64),
    }
    out = session.run(None, inputs)
    embeddings = _normalize(_mean_pool(out[0], enc["attention_mask"]))
    return EmbedResponse(embeddings=embeddings.tolist())
