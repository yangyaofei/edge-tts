# Qwen3-TTS PyTorch Server (faster-qwen3-tts + CUDA Graph)
#
# RTF ~1.49 on GTX 1080 Ti with BF16 + CUDA Graph
# Replaces the C++ qwen3-tts-server with a Python equivalent that uses
# PyTorch + faster-qwen3-tts for kernel fusion and CUDA Graph acceleration.

FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        libsndfile1 \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m ensurepip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Create pyproject.toml and install dependencies
COPY pyproject.toml .
RUN uv venv && uv sync

# Copy server code
COPY server.py .

# Generate a minimal reference audio for speaker embedding extraction
RUN uv run python -c "import numpy as np, wave; sr=24000; t=np.linspace(0,3,int(sr*3),endpoint=False); a=(np.sin(2*np.pi*220*t)*0.3*32767).astype(np.int16); w=wave.open('ref_audio.wav','w'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(a.tobytes()); w.close(); print('ref_audio.wav ok')"

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
