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

import asyncio
import logging
import os
import re
import struct
import threading
import time
from collections.abc import Generator
from contextlib import asynccontextmanager, closing
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np
import torch
from fastapi import FastAPI, File, Request, UploadFile
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


def normalize_language(lang: str | None) -> str:
    if not lang:
        return "chinese"
    k = lang.strip().lower()
    return _LANG_ALIAS.get(k, k)


def split_sentences(text: str, min_len: int = 4) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    out: list[str] = []
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


def wav_header(sr: int, channels: int = 1, bits: int = 16, n_frames: int | None = None) -> bytes:
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
    """Time-stretch: speed>1 = faster, pitch preserved. Uses WSOLA (audiotsm)."""
    if abs(speed - 1.0) < 0.01:
        return pcm
    from audiotsm import wsola
    from audiotsm.io.array import ArrayReader, ArrayWriter
    reader = ArrayReader(pcm.reshape(1, -1).astype(np.float32))
    writer = ArrayWriter(channels=1)
    wsola(reader.channels, speed=speed).run(reader, writer)
    return writer.data[0]


class StreamingTSM:
    """Streaming time-scale modification via WSOLA with input overlap.

    Purpose: produce artifact-free speed change while streaming, by ensuring
    WSOLA has overlap context at every batch boundary.

    Strategy:
      1. Accumulate incoming PCM until a batch threshold is reached.
      2. Prepend the **input tail** from the previous batch (overlap region,
         ~50 ms).  This gives WSOLA continuous context so the first frame
         of each batch can find a valid overlap match.
      3. Run WSOLA on the combined data with a **fresh** WSOLA object
         (no stale internal state).
      4. Discard the output samples corresponding to the overlap region.
      5. Emit the remaining output — it connects smoothly to the previous
         batch because WSOLA processed continuous data across the boundary.

    The overlap is 50 ms (≈1 200 samples at 24 kHz), well above WSOLA's
    default frame_size (512) + search_area (256) = 768 samples.  Only a
    few dozen ms of latency is added per batch.
    """

    def __init__(self, speed: float, sample_rate: int = 24000,
                 buffer_secs: float = 2.0, overlap_ms: float = 100):
        self._speed = speed
        self._sr = sample_rate
        self._batch_samples = int(sample_rate * buffer_secs)
        self._overlap = int(sample_rate * overlap_ms / 1000)
        self._buf: list[np.ndarray] = []
        self._buf_len = 0
        self._prev_tail: np.ndarray | None = None
        self._active = abs(speed - 1.0) >= 0.01
        if self._active:
            from audiotsm import wsola
            from audiotsm.io.array import ArrayReader, ArrayWriter
            self._wsola_cls = wsola
            self._ArrayReader = ArrayReader
            self._ArrayWriter = ArrayWriter

    def _wsola_once(self, data: np.ndarray) -> np.ndarray:
        """Run a fresh WSOLA object on *data* and return the output."""
        tsm = self._wsola_cls(1, speed=self._speed)
        reader = self._ArrayReader(data.reshape(1, -1).astype(np.float32))
        writer = self._ArrayWriter(channels=1)
        tsm.run(reader, writer)
        out = writer.data[0]
        return out if out is not None and len(out) > 0 else np.zeros(0, dtype=np.float32)

    def process(self, pcm: np.ndarray) -> np.ndarray:
        """Accumulate PCM; run WSOLA when buffer reaches batch threshold."""
        if not self._active:
            return pcm
        if len(pcm) == 0:
            return np.zeros(0, dtype=np.float32)

        self._buf.append(pcm)
        self._buf_len += len(pcm)

        if self._buf_len < self._batch_samples:
            return np.zeros(0, dtype=np.float32)

        # --- extract accumulated data ---
        data = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
        self._buf = []
        self._buf_len = 0

        # --- save input tail for next batch's overlap ---
        tail = data[-self._overlap:] if len(data) > self._overlap else data
        tail = tail.copy()

        # --- prepend previous overlap ---
        prev_len = len(self._prev_tail) if self._prev_tail is not None else 0
        if self._prev_tail is not None:
            full = np.concatenate([self._prev_tail, data])
        else:
            full = data
        self._prev_tail = tail

        # --- run WSOLA on continuous data ---
        output = self._wsola_once(full)

        # --- discard head (overlap region + startup artifacts) ---
        if prev_len > 0 and len(output) > 1:
            skip = min(int(prev_len / self._speed), len(output) - 1)
            output = output[skip:]

        return output

    def flush(self) -> np.ndarray:
        """Process any remaining buffered data."""
        if not self._active or self._buf_len == 0:
            self._buf.clear()
            self._buf_len = 0
            return np.zeros(0, dtype=np.float32)

        data = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
        self._buf.clear()
        self._buf_len = 0

        prev_len = len(self._prev_tail) if self._prev_tail is not None else 0
        if self._prev_tail is not None:
            full = np.concatenate([self._prev_tail, data])
        else:
            full = data
        self._prev_tail = None

        output = self._wsola_once(full)
        if prev_len > 0 and len(output) > 1:
            skip = min(int(prev_len / self._speed), len(output) - 1)
            output = output[skip:]
        return output


def make_streaming_tsm(speed: float, sample_rate: int = 24000):
    """Factory: prefer Sonic (true streaming, artifact-free), fall back to
    audiotsm-based StreamingTSM if the native lib is unavailable.

    Both expose the same interface: process(pcm)->ndarray, flush()->ndarray.
    Sonic eliminates batch-boundary click/pop (single stateful stream); the
    audiotsm fallback keeps the old overlap-hack behavior.
    """
    if abs(speed - 1.0) < 0.01:
        return None
    try:
        from tsm_sonic import SonicTSM
        tsm = SonicTSM(speed, sample_rate=sample_rate)
        if tsm.available:
            log.info("streaming TSM: Sonic backend (speed=%.2f)", speed)
            return tsm
        log.warning("streaming TSM: Sonic unavailable, falling back to audiotsm WSOLA")
    except Exception as e:  # noqa: BLE001
        log.warning("streaming TSM: Sonic import failed (%s), falling back to audiotsm", e)
    return StreamingTSM(speed, sample_rate=sample_rate)


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
        except Exception:  # noqa: BLE001
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
        """Load all reference audio files from REF_AUDIO_DIR.

        For each voice, looks for a matching .txt file for ref_text.
        In ICL mode (xvec_only=False), ref_text is required.
        """
        REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        for wav_file in REF_AUDIO_DIR.glob("*.wav"):
            voice_id = wav_file.stem
            if voice_id == "default":
                continue
            txt_file = wav_file.with_suffix(".txt")
            if txt_file.exists():
                ref_text = txt_file.read_text(encoding="utf-8").strip()
            else:
                ref_text = "" if XVEC_ONLY else REF_TEXT
            prompt = self._compute_prompt(str(wav_file), ref_text)
            if prompt is None:
                log.warning("skipped voice %s: prompt computation failed", voice_id)
                continue
            self.ref_audios[voice_id] = {
                "name": voice_id,
                "path": str(wav_file),
                "prompt": prompt,
            }
            log.info("loaded voice: %s from %s (ref_text=%d chars)",
                     voice_id, wav_file, len(ref_text))

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
        except Exception as e:  # noqa: BLE001
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
        except Exception as e:  # noqa: BLE001
            log.warning("warmup failed: %s", e)

    def _build_kwargs(self, text: str, language: str, streaming: bool,
                      voice: str = "default", temperature: float = 0.9,
                      instruct: str | None = None) -> dict:
        kw = {
            "text": text, "language": language, "xvec_only": XVEC_ONLY,
            "max_new_tokens": MAX_NEW_TOKENS,
            "ref_text": "" if XVEC_ONLY else REF_TEXT,
            "temperature": temperature,
            "instruct": instruct,
        }
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
                   temperature: float = 0.9, instruct: str | None = None
                   ) -> Generator[tuple[np.ndarray, int], None, None]:
        try:
            with self.lock:
                for chunk, sr, _t in self.model.generate_voice_clone_streaming(
                    **self._build_kwargs(text, language, streaming=True,
                                         voice=voice, temperature=temperature, instruct=instruct)
                ):
                    yield np.asarray(chunk, dtype=np.float32).reshape(-1), int(sr)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                log.debug("gen_stream cleanup: CUDA cache emptied")

    def gen_full(self, text: str, language: str, voice: str = "default",
                 temperature: float = 0.9, instruct: str | None = None,
                 speed: float = 1.0, pitch: float = 0.0, volume: float = 1.0
                 ) -> tuple[np.ndarray, int]:
        try:
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
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                log.debug("gen_full cleanup: CUDA cache emptied")


M = TTSModel()


# --------------------------------------------------------------------------- #
# FastAPI
# --------------------------------------------------------------------------- #
class SynthReq(BaseModel):
    text: str
    language: str | None = "chinese"
    voice: str | None = "default"
    temperature: float | None = 0.9
    speed: float | None = 1.0          # 1.0=原速, 1.5=快1.5倍, 0.8=慢
    pitch: float | None = 0.0          # 半音, +2=升2半音, -3=降3半音
    volume: float | None = 1.0         # 1.0=原音量, 1.5=+50%, 0.5=减半
    instruct: str | None = None        # 仅 1.7B CustomVoice 有效
    max_new_tokens: int | None = None


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
async def upload_ref_audio(voice_id: str, name: str = "", file: UploadFile = File(...)):  # noqa: B008
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
    log.info("/synthesize voice=%s lang=%s temp=%.1f speed=%.1f pitch=%.1f vol=%.1f gen=%.2fs audio=%.2fs RTF=%.3f%s",
             voice, lang, temperature, speed, pitch, volume, dt, dur, (dur / dt) if dt else 0,
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
async def synthesize_stream(req: SynthReq, request: Request):
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
    log.info("/synthesize_stream voice=%s lang=%s temp=%.1f speed=%.1f pitch=%.1f vol=%.1f sents=%d chars=%d%s",
             voice, lang, temperature, speed, pitch, volume, len(sentences), len(text),
             f" instruct='{instruct[:30]}...'" if instruct else "")

    q: Queue = Queue(maxsize=QUEUE_SIZE)
    cancel_event = threading.Event()
    _DONE = ("done", None)
    _ERR = ("error", None)
    needs_pitch  = abs(pitch) >= 0.01
    needs_speed  = abs(speed - 1.0) >= 0.01
    needs_volume = abs(volume - 1.0) >= 0.01
    needs_batch  = needs_pitch                       # pitch_shift requires full signal
    needs_stream  = (needs_speed or needs_volume) and not needs_batch

    def producer():
        t0 = time.perf_counter()
        total_samples = 0
        sr_out = M.sr
        def _safe_put(item):
            """Put to queue with 10s consumer-dead timeout."""
            deadline = time.monotonic() + 10
            while not cancel_event.is_set():
                if time.monotonic() > deadline:
                    log.warning("producer: queue full 10s, consumer likely dead, aborting")
                    return False
                try:
                    q.put(item, timeout=0.5)
                    return True
                except Full:
                    continue
            return False

        try:
            for i, s in enumerate(sentences):
                if cancel_event.is_set():
                    break
                with closing(M.gen_stream(s, lang, voice=voice,
                                          temperature=temperature, instruct=instruct)) as gen:
                    if needs_batch:
                        # Batch path: pitch_shift needs the complete sentence
                        sent_chunks = []
                        for pcm, sr in gen:
                            if cancel_event.is_set():
                                break
                            sr_out = sr
                            sent_chunks.append(pcm)
                        if sent_chunks and not cancel_event.is_set():
                            full = np.concatenate(sent_chunks)
                            full = post_process(full, sr_out, speed=speed, pitch=pitch, volume=volume)
                            total_samples += len(full)
                            if not _safe_put(("audio", (pcm_to_i16_bytes(full), sr_out))):
                                break
                    elif needs_stream:
                        # Streaming path: WSOLA + volume per chunk
                        tsm = make_streaming_tsm(speed, sample_rate=M.sr) if needs_speed else None
                        for pcm, sr in gen:
                            if cancel_event.is_set():
                                break
                            sr_out = sr
                            if tsm:
                                pcm = tsm.process(pcm)
                            if needs_volume:
                                pcm = volume_change(pcm, volume)
                            if len(pcm) > 0:
                                total_samples += len(pcm)
                                if not _safe_put(("audio", (pcm_to_i16_bytes(pcm), sr))):
                                    break
                        # Drain any remaining WSOLA overlap at sentence end
                        if tsm and not cancel_event.is_set():
                            tail = tsm.flush()
                            if len(tail) > 0:
                                if needs_volume:
                                    tail = volume_change(tail, volume)
                                total_samples += len(tail)
                                _safe_put(("audio", (pcm_to_i16_bytes(tail), sr_out)))
                    else:
                        # Pure passthrough — no DSP
                        for pcm, sr in gen:
                            if cancel_event.is_set():
                                break
                            sr_out = sr
                            total_samples += len(pcm)
                            pcm_bytes = pcm_to_i16_bytes(pcm)
                            if not _safe_put(("audio", (pcm_bytes, sr))):
                                break
            if not cancel_event.is_set():
                q.put(_DONE)
            dt = time.perf_counter() - t0
            log.info("producer done: gen=%.2fs audio=%.2fs RTF=%.3f cancelled=%s",
                     dt, total_samples / sr_out if sr_out else 0,
                     (total_samples / sr_out / dt) if sr_out and dt else 0,
                     cancel_event.is_set())
        except Exception:
            log.exception("producer failed")
            try:
                q.put_nowait(_ERR)
            except Full:
                pass

    threading.Thread(target=producer, daemon=True, name="tts-producer").start()

    async def body():
        loop = asyncio.get_running_loop()
        first = True
        sr = M.sr
        _SILENCE = b"\x00\x00"
        try:
            while True:
                try:
                    item = await loop.run_in_executor(None, lambda: q.get(timeout=1.0))
                except Empty:
                    if first:
                        yield wav_header(sr, n_frames=None)
                        first = False
                    yield _SILENCE
                    if await request.is_disconnected():
                        cancel_event.set()
                        log.info("stream: client disconnected (heartbeat)")
                        return
                    continue
                kind, payload = item
                if kind in ("done", "error"):
                    return
                pcm_bytes, cs = payload
                sr = cs
                if first:
                    yield wav_header(sr, n_frames=None)
                    first = False
                yield pcm_bytes
                if await request.is_disconnected():
                    cancel_event.set()
                    log.info("stream: client disconnected (after chunk)")
                    return
        except GeneratorExit:
            log.info("stream: client disconnected (GeneratorExit)")
            raise
        finally:
            cancel_event.set()

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
