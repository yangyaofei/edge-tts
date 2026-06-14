from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class Qwen3TTSEngine:
    """qwen3-tts-server 的 HTTP 客户端。

    通过 generate_chunk 实现 TTSEngine Protocol。
    支持 ref_audio 声音克隆。
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
        """调用 qwen3-tts-server 生成一段音频。

        Returns:
            WAV bytes
        """
        payload: dict[str, Any] = {
            "text": text,
            "language": self.language,
            "max_tokens": self.max_tokens,
        }

        if ref_audio:
            payload["ref_audio"] = base64.b64encode(ref_audio).decode()

        logger.debug(f"Qwen3TTSEngine: POST {self.server_url}/api/synthesize, {len(text)} chars")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.server_url}/api/synthesize",
                json=payload,
            )
            resp.raise_for_status()
            return resp.content

    async def get_voices(self) -> list[dict[str, Any]]:
        return [
            {"id": "default", "name": "Qwen3-TTS 默认", "language": "zh"},
        ]

    async def health_check(self) -> bool:
        """检查 qwen3-tts-server 是否可用。"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.server_url}/api/health")
                return resp.status_code == 200
        except Exception:
            return False
