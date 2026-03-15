from pathlib import Path

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from config import MODEL_HF_ID

ONNX_DIR = Path(__file__).resolve().parent / "onnx_model"
ONNX_DIR.mkdir(exist_ok=True)

print("Loading model and tokenizer...")
model = ORTModelForFeatureExtraction.from_pretrained(MODEL_HF_ID, export=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_ID)

print(f"Saving to {ONNX_DIR}...")
model.save_pretrained(ONNX_DIR)
tokenizer.save_pretrained(ONNX_DIR)
print("Done.")
