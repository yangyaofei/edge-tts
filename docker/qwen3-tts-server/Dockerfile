# =====================================================================
# qwen3-tts-server — multi-stage Docker build with CUDA support
# =====================================================================

# ---------- Stage 1: builder ----------
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04 AS builder

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

# ---- Build GGML with CUDA (only ggml target, skip tests) ----
RUN cmake -S qwen3-tts.cpp/ggml -B qwen3-tts.cpp/ggml/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=ON \
        -DGGML_TESTS=OFF \
        -DGGML_EXAMPLES=OFF \
    && cmake --build qwen3-tts.cpp/ggml/build -j"$(nproc)" --target ggml

# ---- Copy server sources + our modified qwen3-tts source ----
# Server sources
COPY server.cpp CMakeLists.txt /build/qwen3-tts-server/

# Our modified qwen3-tts.cpp files (streaming support)
COPY tts_transformer.h tts_transformer.cpp qwen3_tts.h qwen3_tts.cpp \
     /build/qwen3-tts.cpp/src/

RUN wget -q -O /build/qwen3-tts-server/httplib.h \
        https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h \
    && wget -q -O /build/qwen3-tts-server/json.hpp \
        https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp

# ---- Build the server ----
RUN cmake -S /build/qwen3-tts-server -B /build/server-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
        -DQWEN3_TTS_SOURCE_DIR=/build/qwen3-tts.cpp \
    && cmake --build /build/server-build -j"$(nproc)" \
              --target qwen3-tts-server

# ---------- Stage 2: runtime ----------
FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy server binary
COPY --from=builder /build/server-build/qwen3-tts-server /app/qwen3-tts-server

# Copy GGML shared libraries (CUDA + CPU backends) with version symlinks
COPY --from=builder /build/qwen3-tts.cpp/ggml/build/src/libggml.so.0.9.6 /app/lib/
COPY --from=builder /build/qwen3-tts.cpp/ggml/build/src/libggml-base.so.0.9.6 /app/lib/
COPY --from=builder /build/qwen3-tts.cpp/ggml/build/src/libggml-cpu.so /app/lib/
COPY --from=builder /build/qwen3-tts.cpp/ggml/build/src/ggml-cuda/libggml-cuda.so /app/lib/

RUN cd /app/lib && \
    for f in *.so.0.9.6; do \
        base="${f%.so.0.9.6}.so"; \
        ln -sf "$f" "${base}.0"; \
        ln -sf "${base}.0" "$base"; \
    done && \
    ln -sf libggml-cpu.so libggml-cpu.so.0 && \
    ln -sf libggml-cuda.so libggml-cuda.so.0 && \
    ls -la

ENV LD_LIBRARY_PATH=/app/lib

VOLUME ["/models"]

EXPOSE 9880

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:9880/api/health || exit 1

ENTRYPOINT ["/app/qwen3-tts-server"]
CMD ["--model-dir", "/models", "--port", "9880"]
