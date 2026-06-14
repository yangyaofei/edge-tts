from __future__ import annotations

import base64
import logging
from typing import Any, AsyncGenerator

import httpx

logger = logging.getLogger(__name__)


class Qwen3TTSEngine:
    """qwen3-tts-server 的 HTTP 客户端。

    通过 generate_chunk 实现 TTSEngine Protocol。
    支持 ref_audio 声音克隆。
    generate_chunk_stream 支持流式接收音频（降低首字延迟）。
    """

    def __init__(
        self,
        server_url: str = "http://localhost:9880",
        language: str = "zh",
        max_tokens: int = 8192,
        timeout: float = 300.0,
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
    ) -> bytes:
        """调用 qwen3-tts-server 生成一段音频，返回完整 WAV bytes。"""
        payload = self._build_payload(text, ref_audio)

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
    ) -> AsyncGenerator[bytes, None]:
        """流式接收音频：server 按句子切分，边生成边返回。

        首字延迟从"整段生成完"降到"首句生成完"。
        """
        payload = self._build_payload(text, ref_audio)

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

    def _build_payload(self, text: str, ref_audio: bytes | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": text,
            "language": self.language,
            "max_tokens": self.max_tokens,
        }
        if ref_audio:
            payload["ref_audio"] = base64.b64encode(ref_audio).decode()
        return payload

    async def get_voices(self) -> list[dict[str, Any]]:
        return [
            {"id": "default", "name": "Qwen3-TTS 默认", "language": "zh"},
        ]

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.server_url}/api/health")
                return resp.status_code == 200
        except Exception:
            return False
