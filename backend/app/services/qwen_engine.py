from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

import httpx

logger = logging.getLogger(__name__)


class Qwen3TTSEngine:
    """qwen3-tts-pytorch server 的 HTTP 客户端。

    通过 faster-qwen3-tts + CUDA Graph 实现推理加速。
    支持声音选择、语速、温度、自然语言指令。
    """

    def __init__(
        self,
        server_url: str = "http://localhost:9880",
        language: str = "zh",
        max_tokens: int = 8192,
        timeout: float = 600.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.language = language
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def generate_chunk(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        ref_audio: bytes | None = None,
        temperature: float | None = None,
        instruct: str | None = None,
        pitch: float = 0.0,
        volume: float = 1.0,
    ) -> bytes:
        """调用 TTS server 生成一段音频，返回完整 WAV bytes。"""
        payload: dict[str, Any] = {
            "text": text,
            "language": self.language,
            "voice": voice,
            "speed": speed,
            "pitch": pitch,
            "volume": volume,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if instruct:
            payload["instruct"] = instruct

        logger.debug(f"Qwen3TTSEngine: POST synthesize, {len(text)} chars, voice={voice}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.server_url}/api/synthesize",
                json=payload,
            )
            resp.raise_for_status()
            return resp.content

    async def generate_chunk_stream(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        ref_audio: bytes | None = None,
        temperature: float | None = None,
        instruct: str | None = None,
        pitch: float = 0.0,
        volume: float = 1.0,
    ) -> AsyncGenerator[bytes, None]:
        """流式合成：server 按句子切分，边生成边返回 PCM。"""
        payload: dict[str, Any] = {
            "text": text,
            "language": self.language,
            "voice": voice,
            "speed": speed,
            "pitch": pitch,
            "volume": volume,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if instruct:
            payload["instruct"] = instruct

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.server_url}/api/synthesize_stream",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    if chunk:
                        yield chunk

    async def get_voices(self) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.server_url}/api/voices")
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("voices", [])
        except Exception:
            pass
        return [{"id": "default", "name": "Qwen3-TTS 默认", "language": "multilingual"}]

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.server_url}/api/health")
                return resp.status_code == 200
        except Exception:
            return False
