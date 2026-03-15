FROM python:3.12-slim

WORKDIR /app

# Install PyTorch CPU-only first so dependencies don't pull in CUDA/cuDNN
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY export_onnx.py .
COPY part1_basic/ part1_basic/
COPY part2_onnx/ part2_onnx/
COPY part3_batching/ part3_batching/
