#!/usr/bin/env python3
"""Qwen3-TTS Python Server with CUDA Graph + paragraph pipeline.

目的 (purpose): 在单 GPU 上提供低首字延迟、高吞吐的 TTS HTTP 服务。
做法 (how):
  - 模型启动时加载一次 + 预捕获 CUDA Graph + 预计算 speaker embedding
  - /api/synthesize        非流式，整段生成后返回完整 WAV
  - /api/synthesize_stream 流式，段落并行 pipeline：producer 线程在 GPU 上
                           连续生成并往有界队列推 PCM，HTTP writer 同时把队列
                           里的 PCM 发出去 → 生成与网络发送重叠

Endpoints:
  GET  /api/health
  POST /api/synthesize         (non-streaming, returns full WAV)
  POST /api/synthesize_stream  (streaming, chunked WAV via pipeline)
"""

from __future__ import annotations

import io
import logging
import os
import re
import struct
import threading
import time
from contextlib import asynccontextmanager
from queue import Queue
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Config (env-overridable)
# --------------------------------------------------------------------------- #
MODEL_ID = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
REF_AUDIO = os.environ.get("TTS_REF_AUDIO", "ref_audio.wav")
REF_TEXT = os.environ.get(
    "TTS_REF_TEXT",
    "I'm confused why some people have super short timelines, yet at the same "
    "time are bullish on scaling up reinforcement learning atop LLMs.",
)
DEVICE = os.environ.get("TTS_DEVICE", "cuda")
DTYPE = torch.bfloat16
MAX_SEQ_LEN = int(os.environ.get("TTS_MAX_SEQ_LEN", "2048"))
CHUNK_SIZE = int(os.environ.get("TTS_CHUNK_SIZE", "8"))      # 8 steps ≈ 667ms/chunk
MAX_NEW_TOKENS = int(os.environ.get("TTS_MAX_NEW_TOKENS", "2048"))
XVEC_ONLY = os.environ.get("TTS_XVEC_ONLY", "1") == "1"      # clean multilingual
QUEUE_SIZE = int(os.environ.get("TTS_QUEUE_SIZE", "32"))     # pipeline backpressure

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s"
)
log = logging.getLogger("tts")

# Language normalization: accept codes + full names → lowercase full name.
_LANG_ALIAS = {
    "zh": "chinese", "cn": "chinese", "zh-cn": "chinese", "zh-tw": "chinese",
    "en": "english", "eng": "english",
    "ja": "japanese", "jp": "japanese",
    "ko": "korean", "kr": "korean",
    "fr": "french", "de": "german", "it": "italian",
    "pt": "portuguese", "ru": "russian", "es": "spanish",
}
_VALID_LANGS = {
    "chinese", "english", "french", "german", "italian",
    "japanese", "korean", "portuguese", "russian", "spanish",
}


def normalize_language(lang: Optional[str]) -> str:
    if not lang:
        return "chinese"
    k = lang.strip().lower()
    return _LANG_ALIAS.get(k, k)


# --------------------------------------------------------------------------- #
# Sentence splitting (CJK + Latin aware)
# 目的: 把长文本切成句，让第一句尽快出第一块音频；后续句与前面句的发送重叠。
# --------------------------------------------------------------------------- #
def split_sentences(text: str, min_len: int = 4) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # split after CJK/latin sentence enders or newlines, keep the delimiter
    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    out: List[str] = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        buf = p if not buf else buf + p
        if len(buf) >= min_len:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out or [text]


# --------------------------------------------------------------------------- #
# WAV helpers
# --------------------------------------------------------------------------- #
def pcm_to_i16_bytes(x: np.ndarray) -> bytes:
    x = np.clip(np.asarray(x, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def wav_header(sr: int, channels: int = 1, bits: int = 16, n_frames: Optional[int] = None) -> bytes:
    """Canonical 44-byte WAV header. n_frames=None → 0x7FFFFFFF (streaming)."""
    byte_rate = sr * channels * bits // 8
    block_align = channels * bits // 8
    if n_frames is None:
        data_sz = 0x7FFFFFFF
    else:
        data_sz = n_frames * block_align
    riff_sz = 36 + data_sz
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_sz & 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16, 1, channels, sr, byte_rate, block_align, bits,
        b"data", data_sz & 0xFFFFFFFF,
    )


def write_full_wav(pcm: np.ndarray, sr: int) -> bytes:
    pcm = np.clip(np.asarray(pcm, dtype=np.float32).reshape(-1), -1.0, 1.0)
    i16 = (pcm * 32767.0).astype("<i2")
    return wav_header(sr, n_frames=len(i16)) + i16.tobytes()


# --------------------------------------------------------------------------- #
# Model wrapper (singleton)
# --------------------------------------------------------------------------- #
class TTSModel:
    def __init__(self) -> None:
        self.model = None
        self.sr: int = 24000
        self.voice_prompt = None
        self.lock = threading.Lock()  # GPU is serial → one generation at a time
        self.ready = False
        self.lang_set = set()

    def load(self) -> None:
        from faster_qwen3_tts import FasterQwen3TTS

        t0 = time.perf_counter()
        self.model = FasterQwen3TTS.from_pretrained(
            MODEL_ID, device=DEVICE, dtype=DTYPE,
            attn_implementation="eager", max_seq_len=MAX_SEQ_LEN,
        )
        self.sr = int(self.model.sample_rate)
        try:
            self.lang_set = set(
                self.model.model.model.config.talker_config.codec_language_id.keys()
            )
        except Exception:
            self.lang_set = _VALID_LANGS
        log.info(
            "model loaded %.1fs  sr=%d  VRAM=%.2fGB  langs=%d",
            time.perf_counter() - t0, self.sr,
            torch.cuda.memory_allocated() / 1e9, len(self.lang_set),
        )
        self._precompute_prompt()
        self._warmup()
        self.ready = True
        log.info("TTSModel READY")

    def _build_kwargs(self, text: str, language: str, streaming: bool) -> dict:
        kw = dict(
            text=text, language=language, xvec_only=XVEC_ONLY,
            max_new_tokens=MAX_NEW_TOKENS, ref_text="" if XVEC_ONLY else REF_TEXT,
        )
        if streaming:
            kw["chunk_size"] = CHUNK_SIZE
        if self.voice_prompt is not None:
            kw["voice_clone_prompt"] = self.voice_prompt
            kw["ref_audio"] = None
        else:
            kw["ref_audio"] = REF_AUDIO
        return kw

    def _precompute_prompt(self) -> None:
        try:
            if XVEC_ONLY:
                items = self.model.model.create_voice_clone_prompt(
                    ref_audio=REF_AUDIO, ref_text="", x_vector_only_mode=True
                )
                spk = items[0].ref_spk_embedding
                self.voice_prompt = {"ref_spk_embedding": [spk]}
            else:
                items = self.model.model.create_voice_clone_prompt(
                    ref_audio=REF_AUDIO, ref_text=REF_TEXT, x_vector_only_mode=False
                )
                self.voice_prompt = items
            log.info("precomputed voice_clone_prompt (xvec_only=%s)", XVEC_ONLY)
        except Exception as e:
            log.warning("precompute prompt failed (%s); fallback to ref_audio per call", e)
            self.voice_prompt = None

    def _warmup(self) -> None:
        # Capture CUDA graphs (predictor + talker) so real requests skip capture.
        try:
            t0 = time.perf_counter()
            n = 0
            for _ in self.model.generate_voice_clone_streaming(
                **self._build_kwargs("warmup.", "english", streaming=True)
            ):
                n += 1
            log.info(
                "warmup ok %.1fs  chunks=%d  peakVRAM=%.2fGB",
                time.perf_counter() - t0, n,
                torch.cuda.max_memory_allocated() / 1e9,
            )
        except Exception as e:
            log.warning("warmup failed: %s", e)

    def gen_stream(self, text: str, language: str) -> Generator[Tuple[np.ndarray, int], None, None]:
        with self.lock:
            for chunk, sr, _t in self.model.generate_voice_clone_streaming(
                **self._build_kwargs(text, language, streaming=True)
            ):
                yield np.asarray(chunk, dtype=np.float32).reshape(-1), int(sr)

    def gen_full(self, text: str, language: str) -> Tuple[np.ndarray, int]:
        with self.lock:
            arrays, sr = self.model.generate_voice_clone(
                **self._build_kwargs(text, language, streaming=False)
            )
            pcm = np.concatenate(
                [np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays]
            ) if arrays else np.zeros(0, dtype=np.float32)
            return pcm, int(sr)


M = TTSModel()


# --------------------------------------------------------------------------- #
# FastAPI
# --------------------------------------------------------------------------- #
class SynthReq(BaseModel):
    text: str
    language: Optional[str] = "chinese"
    max_new_tokens: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    M.load()
    yield


app = FastAPI(title="Qwen3-TTS Server (faster-qwen3-tts + CUDA Graph)", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {
        "ok": M.ready,
        "model": MODEL_ID,
        "sample_rate": M.sr,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 3) if torch.cuda.is_available() else 0,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else 0,
        "xvec_only": XVEC_ONLY,
        "chunk_size": CHUNK_SIZE,
        "languages": sorted(M.lang_set),
    }


@app.post("/api/synthesize")
def synthesize(req: SynthReq):
    if not M.ready:
        return JSONResponse({"error": "model not ready"}, status_code=503)
    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    lang = normalize_language(req.language)
    t0 = time.perf_counter()
    pcm, sr = M.gen_full(text, lang)
    dt = time.perf_counter() - t0
    dur = len(pcm) / sr if sr else 0.0
    wav = write_full_wav(pcm, sr)
    log.info(
        "/synthesize lang=%s gen=%.2fs audio=%.2fs RTF=%.3f",
        lang, dt, dur, (dur / dt) if dt else 0,
    )
    return Response(
        content=wav, media_type="audio/wav",
        headers={
            "X-Gen-Time": f"{dt:.3f}",
            "X-Audio-Duration": f"{dur:.3f}",
            "X-Sample-Rate": str(sr),
            "X-RTF": f"{(dur / dt) if dt else 0:.3f}",
            "Access-Control-Expose-Headers": "*",
        },
    )


@app.post("/api/synthesize_stream")
def synthesize_stream(req: SynthReq):
    if not M.ready:
        return JSONResponse({"error": "model not ready"}, status_code=503)
    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    lang = normalize_language(req.language)
    sentences = split_sentences(text) or [text]
    log.info("/synthesize_stream lang=%s sentences=%d total_chars=%d", lang, len(sentences), len(text))

    q: Queue = Queue(maxsize=QUEUE_SIZE)
    _DONE = ("done", None)
    _ERR = ("error", None)

    # producer: pulls from GPU as fast as it produces → real overlap with network send
    def producer():
        t0 = time.perf_counter()
        total_samples = 0
        sr_out = M.sr
        try:
            for i, s in enumerate(sentences):
                for pcm, sr in M.gen_stream(s, lang):
                    sr_out = sr
                    total_samples += len(pcm)
                    q.put(("audio", (pcm_to_i16_bytes(pcm), sr)))
            q.put(_DONE)
            dt = time.perf_counter() - t0
            log.info(
                "producer done: gen=%.2fs audio=%.2fs RTF=%.3f",
                dt, total_samples / sr_out if sr_out else 0,
                (total_samples / sr_out / dt) if sr_out and dt else 0,
            )
        except Exception as e:
            log.exception("producer failed")
            q.put(_ERR)

    threading.Thread(target=producer, daemon=True, name="tts-producer").start()

    # consumer: HTTP writer — sends WAV header then streams PCM
    def body():
        first = True
        sr = M.sr
        while True:
            kind, payload = q.get()
            if kind == "done":
                return
            if kind == "error":
                return
            pcm_bytes, cs = payload
            sr = cs
            if first:
                yield wav_header(sr, n_frames=None)  # unknown size (streaming WAV)
                first = False
            yield pcm_bytes

    return StreamingResponse(
        body(), media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(M.sr),
            "X-Languages": ",".join(sorted(M.lang_set)),
            "Access-Control-Expose-Headers": "*",
            "Cache-Control": "no-cache",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "9881")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
