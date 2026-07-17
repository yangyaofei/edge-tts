#!/usr/bin/env python3
"""Sonic-based streaming Time-Scale Modification (TSM).

目的 (purpose): 为 TTS 提供无批次边界伪影的流式变速。
做法 (how):
  - 包装 Bill Cox 的 Sonic 纯 C 库 (PICOLA-like, 专为语音变速设计)
  - 通过 ctypes 暴露有状态的 writeFloat/readFloat 流式 API
  - 单个 sonicStream 贯穿整条 TTS 音频流 → 跨块状态连续, 无 click/pop

对比 audiotsm (批处理 WSOLA, 每块独立 run → 边界 click):
  实测 (sine 440Hz, sr=24000, speed=1.5, chunk=8000):
    audiotsm_chunked : click_ratio 8.17x, 长度比 0.768 (严重失真)
    sonic_stream     : click_ratio 1.00x, 长度比 0.997 (无伪影)
    首字节延迟: 738 样本 (30.8ms)

接口与原 StreamingTSM 完全一致: process(pcm) / flush() / speed==1.0 直通。
"""
from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("tts")

# --------------------------------------------------------------------------- #
# native lib loader
# --------------------------------------------------------------------------- #
def _find_libsonic() -> str | None:
    candidates = []
    env = os.environ.get("TSM_SONIC_LIB")
    if env:
        candidates.append(env)
    here = Path(__file__).resolve().parent
    candidates += [
        here / "libsonic.so",
        here / "native" / "libsonic.so",
        Path("/app/libsonic.so"),
        Path("/app/tsm_test/libsonic.so"),
    ]
    for c in candidates:
        c = Path(c)
        if c.is_file():
            return str(c)
    return None


_LIBSONIC: ctypes.CDLL | None = None
_LIBSONIC_PATH: str | None = None


def _load_libsonic() -> ctypes.CDLL | None:
    global _LIBSONIC, _LIBSONIC_PATH
    if _LIBSONIC is not None:
        return _LIBSONIC
    path = _find_libsonic()
    if path is None:
        log.warning("SonicTSM: libsonic.so not found, will fall back to audiotsm")
        return None
    try:
        lib = ctypes.CDLL(path)
        lib.sonicCreateStream.restype = ctypes.c_void_p
        lib.sonicCreateStream.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.sonicSetSpeed.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.sonicWriteFloatToStream.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.sonicWriteFloatToStream.restype = ctypes.c_int
        lib.sonicReadFloatFromStream.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.sonicReadFloatFromStream.restype = ctypes.c_int
        lib.sonicSamplesAvailable.argtypes = [ctypes.c_void_p]
        lib.sonicSamplesAvailable.restype = ctypes.c_int
        lib.sonicFlushStream.argtypes = [ctypes.c_void_p]
        lib.sonicFlushStream.restype = ctypes.c_int
        lib.sonicDestroyStream.argtypes = [ctypes.c_void_p]
        _LIBSONIC = lib
        _LIBSONIC_PATH = path
        log.info("SonicTSM: loaded %s", path)
        return lib
    except OSError as e:
        log.warning("SonicTSM: failed to load %s: %s", path, e)
        return None


# --------------------------------------------------------------------------- #
# Streaming TSM (drop-in replacement for audiotsm-based StreamingTSM)
# --------------------------------------------------------------------------- #
class SonicTSM:
    """Stateful streaming time-stretch via the Sonic C library.

    Lifecycle mirrors the existing StreamingTSM:
        tsm = SonicTSM(speed, sample_rate)
        for chunk in tts_stream:
            out = tsm.process(chunk)   # may be empty while priming
            send(out)
        tail = tsm.flush()            # drain residual
        send(tail)
    """

    def __init__(self, speed: float, sample_rate: int = 24000):
        self._speed = float(speed)
        self._sr = int(sample_rate)
        self._active = abs(self._speed - 1.0) >= 0.01
        self._lib: ctypes.CDLL | None = None
        self._h: int | None = None
        if self._active:
            lib = _load_libsonic()
            if lib is not None:
                self._lib = lib
                self._h = lib.sonicCreateStream(self._sr, 1)
                if not self._h:
                    self._lib = None
                    log.warning("SonicTSM: sonicCreateStream failed")
                else:
                    lib.sonicSetSpeed(self._h, ctypes.c_float(self._speed))

    @property
    def available(self) -> bool:
        """True if the Sonic backend is ready (else caller should fall back)."""
        return self._h is not None

    def _read_all(self) -> np.ndarray:
        lib, h = self._lib, self._h
        parts: list[np.ndarray] = []
        while True:
            ready = lib.sonicSamplesAvailable(h)
            if ready <= 0:
                break
            buf = np.zeros(ready, dtype=np.float32)
            lib.sonicReadFloatFromStream(
                h, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ready)
            parts.append(buf)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def process(self, pcm: np.ndarray) -> np.ndarray:
        """Feed one chunk, return all currently-ready output samples.

        Returns an empty array while Sonic primes (~30ms / 738 samples of
        accumulated input). Output is continuous across calls — there are no
        per-chunk boundary artifacts because a single stream holds all state.
        """
        if not self._active:
            return pcm.astype(np.float32, copy=False)
        if self._h is None or len(pcm) == 0:
            return np.zeros(0, dtype=np.float32)
        s = np.ascontiguousarray(pcm.reshape(-1), dtype=np.float32)
        self._lib.sonicWriteFloatToStream(
            self._h, s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(s))
        return self._read_all()

    def flush(self) -> np.ndarray:
        """Flush the internal pipeline and return any residual output."""
        if not self._active or self._h is None:
            return np.zeros(0, dtype=np.float32)
        self._lib.sonicFlushStream(self._h)
        out = self._read_all()
        self._destroy()
        return out

    def _destroy(self):
        if self._h is not None and self._lib is not None:
            try:
                self._lib.sonicDestroyStream(self._h)
            except Exception:
                pass
        self._h = None
        self._lib = None

    def __del__(self):
        self._destroy()
