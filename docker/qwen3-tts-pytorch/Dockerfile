# Qwen3-TTS PyTorch Server (faster-qwen3-tts + CUDA Graph)
#
# RTF ~1.49 on GTX 1080 Ti with BF16 + CUDA Graph
# Replaces the C++ qwen3-tts-server with a Python equivalent that uses
# PyTorch + faster-qwen3-tts for kernel fusion and CUDA Graph acceleration.

FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        curl \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Create pyproject.toml and install dependencies
COPY pyproject.toml .
RUN uv venv && uv sync

# Copy server code
COPY server.py .

# Model cache directory (mount or auto-download)
ENV HF_HOME=/models/hf_cache
ENV TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
ENV TTS_DEVICE=cuda
ENV PORT=9880

VOLUME ["/models"]

EXPOSE 9880

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:9880/api/health || exit 1

CMD ["uv", "run", "python", "server.py"]
