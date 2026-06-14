from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

import edge_tts

logger = logging.getLogger(__name__)


class EdgeTTSEngine:
    """Microsoft Edge TTS engine。

    通过 generate_chunk 实现 TTSEngine Protocol。
    ref_audio 参数被忽略（Edge 不支持声音克隆）。
    """

    def __init__(self) -> None:
        pass

    async def generate_chunk(
        self,
        text: str,
        voice: str = "zh-CN-XiaoxiaoNeural",
        speed: float = 1.0,
        ref_audio: bytes | None = None,
    ) -> bytes:
        rate_str = f"+{int((speed - 1.0) * 100)}%" if speed >= 1.0 else f"{int((speed - 1.0) * 100)}%"
        pitch_str = "+0Hz"

        communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch_str)

        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])

        logger.debug(f"EdgeTTSEngine: {len(text)} chars → {sum(len(c) for c in chunks)} bytes")
        return b"".join(chunks)

    async def generate_stream(
        self,
        text: str,
        voice: str = "zh-CN-XiaoxiaoNeural",
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AsyncGenerator[bytes, None]:
        """流式生成（兼容旧 API）。"""
        rate = kwargs.get("rate")
        if rate is None:
            rate = f"+{int((speed - 1.0) * 100)}%" if speed >= 1.0 else f"{int((speed - 1.0) * 100)}%"
        pitch = kwargs.get("pitch", "+0Hz")

        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    @staticmethod
    async def get_voices() -> list[dict[str, Any]]:
        voices = await edge_tts.list_voices()
        return voices
