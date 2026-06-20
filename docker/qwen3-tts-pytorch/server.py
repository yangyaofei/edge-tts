#!/usr/bin/env python3
"""Qwen3-TTS Python Server with CUDA Graph + paragraph pipeline.

目的 (purpose): 在单 GPU 上提供低首字延迟、高吞吐的 TTS HTTP 服务。
做法 (how):
  - 模型启动时加载一次 + 预捕获 CUDA Graph + 预计算 speaker embedding
  - /api/synthesize        非流式，整段生成后返回完整 WAV
  - /api/synthesize_stream 流式，段落并行 pipeline
  - /api/voices            列出可用声音（参考音频）
  - POST /api/ref_audio    上传自定义参考音频（声音克隆）

Endpoints:
  GET  /api/health
  GET  /api/voices
  POST /api/synthesize         (non-streaming, returns full WAV)
  POST /api/synthesize_stream  (streaming, chunked WAV via pipeline)
  POST /api/ref_audio          (upload reference audio for voice cloning)
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
from pathlib import Path
from queue import Queue
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
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
CHUNK_SIZE = int(os.environ.get("TTS_CHUNK_SIZE", "8"))
MAX_NEW_TOKENS = int(os.environ.get("TTS_MAX_NEW_TOKENS", "2048"))
XVEC_ONLY = os.environ.get("TTS_XVEC_ONLY", "1") == "1"
QUEUE_SIZE = int(os.environ.get("TTS_QUEUE_SIZE", "32"))
REF_AUDIO_DIR = Path(os.environ.get("TTS_REF_AUDIO_DIR", "/app/ref_audios"))

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


def split_sentences(text: str, min_len: int = 4) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
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
    byte_rate = sr * channels * bits // 8
    block_align = channels * bits // 8
    data_sz = 0x7FFFFFFF if n_frames is None else n_frames * block_align
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


def speed_change(pcm: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Time-stretch: speed>1 = faster, pitch preserved. Uses librosa WSOLA."""
    if abs(speed - 1.0) < 0.01:
        return pcm
    import librosa
    return librosa.effects.time_stretch(pcm.astype(np.float32), rate=1.0 / speed)


def pitch_change(pcm: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    """Pitch-shift: n_semitones>0 = higher, tempo preserved. Uses librosa."""
    if abs(n_semitones) < 0.01:
        return pcm
    import librosa
    return librosa.effects.pitch_shift(pcm.astype(np.float32), sr=sr, n_steps=n_semitones)


def volume_change(pcm: np.ndarray, volume: float) -> np.ndarray:
    """Volume scale: 1.0=original, 1.5=+50%, 0.5=half."""
    if abs(volume - 1.0) < 0.01:
        return pcm
    return np.clip(pcm * volume, -1.0, 1.0)


def post_process(pcm: np.ndarray, sr: int, speed: float = 1.0,
                 pitch: float = 0.0, volume: float = 1.0) -> np.ndarray:
    """Apply DSP post-processing: pitch_shift → time_stretch → volume."""
    if pitch != 0.0:
        pcm = pitch_change(pcm, sr, pitch)
    if speed != 1.0:
        pcm = speed_change(pcm, sr, speed)
    if volume != 1.0:
        pcm = volume_change(pcm, volume)
    return pcm


# --------------------------------------------------------------------------- #
# Model wrapper (singleton)
# --------------------------------------------------------------------------- #
class TTSModel:
    def __init__(self) -> None:
        self.model = None
        self.sr: int = 24000
        self.voice_prompt = None
        self.lock = threading.Lock()
        self.ready = False
        self.lang_set = set()
        self.ref_audios: dict[str, dict] = {}  # voice_id → {path, prompt, name}

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
        self._register_default_voice()
        self._load_ref_audios()
        self._warmup()
        self.ready = True
        log.info("TTSModel READY  voices=%s", list(self.ref_audios.keys()))

    def _register_default_voice(self) -> None:
        """Register the default reference audio as voice 'default'."""
        prompt = self._compute_prompt(REF_AUDIO, REF_TEXT)
        self.ref_audios["default"] = {
            "name": "默认声音",
            "path": REF_AUDIO,
            "prompt": prompt,
        }

    def _load_ref_audios(self) -> None:
        """Load all reference audio files from REF_AUDIO_DIR."""
        REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        for wav_file in REF_AUDIO_DIR.glob("*.wav"):
            voice_id = wav_file.stem
            if voice_id == "default":
                continue
            prompt = self._compute_prompt(str(wav_file), "")
            self.ref_audios[voice_id] = {
                "name": voice_id,
                "path": str(wav_file),
                "prompt": prompt,
            }
            log.info("loaded voice: %s from %s", voice_id, wav_file)

    def register_voice(self, voice_id: str, wav_path: str, name: str = "") -> bool:
        """Register a new voice from an audio file."""
        prompt = self._compute_prompt(wav_path, "")
        if prompt is None:
            return False
        self.ref_audios[voice_id] = {
            "name": name or voice_id,
            "path": wav_path,
            "prompt": prompt,
        }
        log.info("registered voice: %s", voice_id)
        return True

    def _compute_prompt(self, ref_audio: str, ref_text: str):
        try:
            if XVEC_ONLY:
                items = self.model.model.create_voice_clone_prompt(
                    ref_audio=ref_audio, ref_text="", x_vector_only_mode=True
                )
                spk = items[0].ref_spk_embedding
                return {"ref_spk_embedding": [spk]}
            else:
                items = self.model.model.create_voice_clone_prompt(
                    ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=False
                )
                return items
        except Exception as e:
            log.warning("compute prompt failed for %s: %s", ref_audio, e)
            return None

    def _warmup(self) -> None:
        try:
            t0 = time.perf_counter()
            n = 0
            for _ in self.model.generate_voice_clone_streaming(
                **self._build_kwargs("warmup.", "english", streaming=True,
                                     voice="default", temperature=0.9, instruct=None)
            ):
                n += 1
            log.info("warmup ok %.1fs  chunks=%d  peakVRAM=%.2fGB",
                     time.perf_counter() - t0, n,
                     torch.cuda.max_memory_allocated() / 1e9)
        except Exception as e:
            log.warning("warmup failed: %s", e)

    def _build_kwargs(self, text: str, language: str, streaming: bool,
                      voice: str = "default", temperature: float = 0.9,
                      instruct: Optional[str] = None) -> dict:
        kw = dict(
            text=text, language=language, xvec_only=XVEC_ONLY,
            max_new_tokens=MAX_NEW_TOKENS,
            ref_text="" if XVEC_ONLY else REF_TEXT,
            temperature=temperature,
            instruct=instruct,
        )
        if streaming:
            kw["chunk_size"] = CHUNK_SIZE

        voice_data = self.ref_audios.get(voice) or self.ref_audios.get("default")
        if voice_data and voice_data.get("prompt") is not None:
            kw["voice_clone_prompt"] = voice_data["prompt"]
            kw["ref_audio"] = None
        else:
            kw["ref_audio"] = REF_AUDIO
        return kw

    def gen_stream(self, text: str, language: str, voice: str = "default",
                   temperature: float = 0.9, instruct: Optional[str] = None
                   ) -> Generator[Tuple[np.ndarray, int], None, None]:
        with self.lock:
            for chunk, sr, _t in self.model.generate_voice_clone_streaming(
                **self._build_kwargs(text, language, streaming=True,
                                     voice=voice, temperature=temperature, instruct=instruct)
            ):
                yield np.asarray(chunk, dtype=np.float32).reshape(-1), int(sr)

    def gen_full(self, text: str, language: str, voice: str = "default",
                 temperature: float = 0.9, instruct: Optional[str] = None,
                 speed: float = 1.0, pitch: float = 0.0, volume: float = 1.0
                 ) -> Tuple[np.ndarray, int]:
        with self.lock:
            arrays, sr = self.model.generate_voice_clone(
                **self._build_kwargs(text, language, streaming=False,
                                     voice=voice, temperature=temperature, instruct=instruct)
            )
            pcm = np.concatenate(
                [np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays]
            ) if arrays else np.zeros(0, dtype=np.float32)
            pcm = post_process(pcm, sr, speed=speed, pitch=pitch, volume=volume)
            return pcm, int(sr)


M = TTSModel()


# --------------------------------------------------------------------------- #
# FastAPI
# --------------------------------------------------------------------------- #
class SynthReq(BaseModel):
    text: str
    language: Optional[str] = "chinese"
    voice: Optional[str] = "default"
    temperature: Optional[float] = 0.9
    speed: Optional[float] = 1.0          # 1.0=原速, 1.5=快1.5倍, 0.8=慢
    pitch: Optional[float] = 0.0          # 半音, +2=升2半音, -3=降3半音
    volume: Optional[float] = 1.0         # 1.0=原音量, 1.5=+50%, 0.5=减半
    instruct: Optional[str] = None        # 仅 1.7B CustomVoice 有效
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
        "voices": [{"id": vid, "name": v["name"]} for vid, v in M.ref_audios.items()],
    }


@app.get("/api/voices")
def get_voices():
    return {
        "voices": [
            {"id": vid, "name": v["name"], "language": "multilingual"}
            for vid, v in M.ref_audios.items()
        ]
    }


@app.post("/api/ref_audio/{voice_id}")
async def upload_ref_audio(voice_id: str, name: str = "", file: UploadFile = File(...)):
    """Upload a reference audio file to register a new voice."""
    REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    save_path = REF_AUDIO_DIR / f"{voice_id}.wav"
    content = await file.read()
    save_path.write_bytes(content)
    ok = M.register_voice(voice_id, str(save_path), name)
    if ok:
        return {"ok": True, "voice_id": voice_id, "name": name or voice_id}
    return JSONResponse({"error": "Failed to process reference audio"}, status_code=400)


@app.post("/api/synthesize")
def synthesize(req: SynthReq):
    if not M.ready:
        return JSONResponse({"error": "model not ready"}, status_code=503)
    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    lang = normalize_language(req.language)
    voice = req.voice or "default"
    temperature = req.temperature if req.temperature is not None else 0.9
    speed = req.speed if req.speed is not None else 1.0
    pitch = req.pitch if req.pitch is not None else 0.0
    volume = req.volume if req.volume is not None else 1.0
    instruct = req.instruct

    t0 = time.perf_counter()
    pcm, sr = M.gen_full(text, lang, voice=voice, temperature=temperature,
                         instruct=instruct, speed=speed, pitch=pitch, volume=volume)
    dt = time.perf_counter() - t0
    dur = len(pcm) / sr if sr else 0.0
    wav = write_full_wav(pcm, sr)
    log.info("/synthesize voice=%s lang=%s temp=%.1f speed=%.1f gen=%.2fs audio=%.2fs RTF=%.3f%s",
             voice, lang, temperature, speed, dt, dur, (dur / dt) if dt else 0,
             f" instruct='{instruct[:30]}...'" if instruct else "")
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
    voice = req.voice or "default"
    temperature = req.temperature if req.temperature is not None else 0.9
    speed = req.speed if req.speed is not None else 1.0
    pitch = req.pitch if req.pitch is not None else 0.0
    volume = req.volume if req.volume is not None else 1.0
    instruct = req.instruct
    sentences = split_sentences(text) or [text]
    log.info("/synthesize_stream voice=%s lang=%s temp=%.1f speed=%.1f pitch=%.1f sentences=%d chars=%d%s",
             voice, lang, temperature, speed, len(sentences), len(text),
             f" instruct='{instruct[:30]}...'" if instruct else "")

    q: Queue = Queue(maxsize=QUEUE_SIZE)
    _DONE = ("done", None)
    _ERR = ("error", None)

    def producer():
        t0 = time.perf_counter()
        total_samples = 0
        sr_out = M.sr
        try:
            for i, s in enumerate(sentences):
                for pcm, sr in M.gen_stream(s, lang, voice=voice,
                                            temperature=temperature, instruct=instruct):
                    sr_out = sr
                    pcm = post_process(pcm, sr, speed=speed, pitch=pitch, volume=volume)
                    total_samples += len(pcm)
                    q.put(("audio", (pcm_to_i16_bytes(pcm), sr)))
            q.put(_DONE)
            dt = time.perf_counter() - t0
            log.info("producer done: gen=%.2fs audio=%.2fs RTF=%.3f",
                     dt, total_samples / sr_out if sr_out else 0,
                     (total_samples / sr_out / dt) if sr_out and dt else 0)
        except Exception as e:
            log.exception("producer failed")
            q.put(_ERR)

    threading.Thread(target=producer, daemon=True, name="tts-producer").start()

    def body():
        first = True
        sr = M.sr
        while True:
            kind, payload = q.get()
            if kind in ("done", "error"):
                return
            pcm_bytes, cs = payload
            sr = cs
            if first:
                yield wav_header(sr, n_frames=None)
                first = False
            yield pcm_bytes

    return StreamingResponse(
        body(), media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(M.sr),
            "X-Voices": ",".join(M.ref_audios.keys()),
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
