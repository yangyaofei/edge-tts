# =====================================================================
# qwen3-tts-server — multi-stage Docker build
#
# Stage 1 (builder):  clone + build qwen3-tts.cpp (with GGML), then
#                     compile the HTTP server that links against it.
# Stage 2 (runtime):  slim image carrying only the server binary.
# =====================================================================

# ---------- Stage 1: builder ----------
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ---- Clone qwen3-tts.cpp (with GGML submodule) ----
ARG QWEN3_TTS_REPO=https://github.com/predict-woo/qwen3-tts.cpp.git
ARG QWEN3_TTS_REF=main

RUN git clone --recurse-submodules --depth 1 \
        -b "${QWEN3_TTS_REF}" "${QWEN3_TTS_REPO}" qwen3-tts.cpp

# ---- Build GGML first (vendored under qwen3-tts.cpp/ggml) ----
# CPU backend only inside Docker (no Metal/CUDA in the base image).
# Static libs so the server binary is self-contained in the runtime stage.
RUN cmake -S qwen3-tts.cpp/ggml -B qwen3-tts.cpp/ggml/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_STATIC=ON \
        -DGGML_NATIVE=ON \
    && cmake --build qwen3-tts.cpp/ggml/build -j"$(nproc)"

# ---- Copy server sources + download header-only deps ----
COPY server.cpp CMakeLists.txt /build/qwen3-tts-server/

RUN wget -q -O /build/qwen3-tts-server/httplib.h \
        https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h \
    && wget -q -O /build/qwen3-tts-server/json.hpp \
        https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp

# ---- Configure & build the server ----
# The server's CMakeLists adds qwen3-tts.cpp as a subdirectory, so all
# static libraries are built and linked transitively.
RUN cmake -S /build/qwen3-tts-server -B /build/server-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DQWEN3_TTS_SOURCE_DIR=/build/qwen3-tts.cpp \
    && cmake --build /build/server-build -j"$(nproc)" \
              --target qwen3-tts-server

# ---------- Stage 2: runtime ----------
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Minimal runtime deps (libstdc++ comes from the base image; libgomp
# is needed for OpenMP used by GGML's CPU backend).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the server binary from the builder stage
COPY --from=builder /build/server-build/qwen3-tts-server /app/qwen3-tts-server

# Models are expected at /models — mount them at runtime
# Required files:
#   /models/qwen3-tts-0.6b-q8_0.gguf   (or -f16.gguf)
#   /models/qwen3-tts-tokenizer-f16.gguf
VOLUME ["/models"]

EXPOSE 9880

# Health check (requires curl; lightweight enough to keep)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:9880/api/health || exit 1

ENTRYPOINT ["/app/qwen3-tts-server"]
CMD ["--model-dir", "/models", "--port", "9880"]
