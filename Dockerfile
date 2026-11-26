# Stage 1: Build the React + Vite frontend bundle
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


# Stage 2: Python 3.13 + PyTorch + Ultralytics + FastAPI inference runtime
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# Runtime libraries for OpenCV and HEIC image decoding (pillow-heif)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        libheif1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Default to CPU PyTorch wheels (lightweight & ideal for Docker Desktop).
# Pass --build-arg TORCH_INDEX_URL="" to install standard PyPI CUDA-enabled wheels.
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

COPY requirements.txt ./
RUN if [ -n "$TORCH_INDEX_URL" ]; then \
         pip install --index-url "$TORCH_INDEX_URL" torch==2.15.0 torchvision==0.29.2; \
       fi \
    && grep -vE '^(ipykernel|torch|torchvision|#|$)' requirements.txt > /tmp/requirements.docker.txt \
    && if [ -z "$TORCH_INDEX_URL" ]; then \
         pip install torch==2.15.0 torchvision==0.29.2; \
       fi \
    && pip install -r /tmp/requirements.docker.txt \
    && rm -f /tmp/requirements.docker.txt

# Application code, configs, available checkpoints (.dockerignore keeps best.pt only), and built frontend
COPY configs/ ./configs/
COPY src/ ./src/
COPY server/ ./server/
COPY runs/ ./runs/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
