from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from config import DEFAULT_PROMPT, MODEL_HF_ID

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_HF_ID)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(title="Embedding API (basic)", lifespan=lifespan)


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
    model = get_model()
    prefixed = [f"{req.prompt}{t}" for t in req.texts]
    embeddings = model.encode(prefixed, convert_to_numpy=True)
    return EmbedResponse(embeddings=embeddings.tolist())
