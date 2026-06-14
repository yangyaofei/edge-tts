from __future__ import annotations

from typing import Protocol, AsyncGenerator, runtime_checkable, Any


@runtime_checkable
class TTSEngine(Protocol):
    """Engine 只负责：干净 chunk 文本 → 音频 bytes。

    不做预处理、不 chunk、不管 ref_audio 状态。
    Pipeline 层负责全部编排。
    """

    async def generate_chunk(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        ref_audio: bytes | None = None,
    ) -> bytes:
        """生成一段音频，返回完整 WAV/MP3 bytes。

        Args:
            text: 已预处理的干净文本（无 Markdown，多音字已处理）
            voice: 声音 ID
            speed: 语速倍率 (1.0 = 正常)
            ref_audio: 参考音频 bytes（用于声音克隆，engine 可选支持）

        Returns:
            完整音频 bytes (WAV/MP3)
        """
        ...

    async def get_voices(self) -> list[dict[str, Any]]:
        """返回可用声音列表。

        Returns:
            [{"id": "...", "name": "...", "language": "..."}]
        """
        ...
