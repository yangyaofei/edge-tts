from __future__ import annotations

import logging
import struct
import wave
import io
from typing import AsyncGenerator

from app.services.base import TTSEngine
from app.services.text_preprocessor import TextPreprocessor
from app.services.polyphone import PolyphoneFixer
from app.services.chunker import TextChunker

logger = logging.getLogger(__name__)


class TTSPipeline:
    """TTS 编排层：预处理 → 多音字 → 分段 → engine 合成。

    所有预处理能力都是可选的（传 None 跳过）。
    Engine 只负责 chunk 文本 → 音频 bytes。
    Pipeline 负责 ref_audio 状态管理和段间静音。
    """

    def __init__(
        self,
        engine: TTSEngine,
        preprocessor: TextPreprocessor | None = None,
        polyphone_fixer: PolyphoneFixer | None = None,
        chunker: TextChunker | None = None,
        ref_trim_seconds: int = 8,
        silence_between_chunks: float = 0.3,
        sample_rate: int = 24000,
    ) -> None:
        self.engine = engine
        self.preprocessor = preprocessor
        self.polyphone_fixer = polyphone_fixer
        self.chunker = chunker
        self.ref_trim_seconds = ref_trim_seconds
        self.silence_between_chunks = silence_between_chunks
        self.sample_rate = sample_rate

    async def generate_stream(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        use_preprocess: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        """完整 pipeline 流式生成。

        Args:
            text: 原始输入文本
            voice: 声音 ID
            speed: 语速倍率
            use_preprocess: 是否执行预处理（可按请求关闭）

        Yields:
            音频 bytes (WAV format)
        """
        processed = text

        # 1. 预处理
        if use_preprocess:
            if self.preprocessor:
                processed = self.preprocessor.process(processed)
            if self.polyphone_fixer:
                processed = self.polyphone_fixer.fix(processed)

        # 2. 分段
        if self.chunker:
            chunks = self.chunker.chunk_text(processed)
        else:
            chunks = [processed] if processed else []

        if not chunks:
            return

        # 3. 逐段生成
        ref_audio: bytes | None = None
        silence_inserted = False

        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            logger.debug(f"Pipeline chunk {i}/{len(chunks)}: {len(chunk_text)} chars")

            audio = await self.engine.generate_chunk(
                chunk_text,
                voice=voice,
                speed=speed,
                ref_audio=ref_audio,
            )

            # 首段提取参考音频
            if i == 0 and len(chunks) > 1:
                ref_audio = self._extract_ref(audio, self.ref_trim_seconds)

            # 段间插入静音（非首段）
            if silence_inserted and self.silence_between_chunks > 0:
                yield self._make_silence(audio, self.silence_between_chunks)
            else:
                yield audio
                silence_inserted = True

    def _extract_ref(self, wav_bytes: bytes, seconds: int) -> bytes:
        """从 WAV bytes 中截取前 N 秒作为参考音频。

        Returns:
            完整的 WAV bytes（含 header），截断后的长度
        """
        try:
            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as wav:
                n_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                framerate = wav.getframerate()
                total_frames = wav.getnframes()

                trim_frames = min(int(seconds * framerate), total_frames)
                frames = wav.readframes(trim_frames)

            out_buf = io.BytesIO()
            with wave.open(out_buf, "wb") as out_wav:
                out_wav.setnchannels(n_channels)
                out_wav.setsampwidth(sample_width)
                out_wav.setframerate(framerate)
                out_wav.writeframes(frames)

            return out_buf.getvalue()
        except Exception as e:
            logger.warning(f"Failed to extract ref audio: {e}, using full audio")
            return wav_bytes

    def _make_silence(self, ref_wav: bytes, seconds: float) -> bytes:
        """构造一段静音 WAV，格式匹配 ref_wav 的 header。

        Args:
            ref_wav: 参考音频（用于获取格式信息）
            seconds: 静音秒数

        Returns:
            静音 WAV bytes
        """
        try:
            buf = io.BytesIO(ref_wav)
            with wave.open(buf, "rb") as wav:
                n_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                framerate = wav.getframerate()

            n_frames = int(seconds * framerate)
            silence_frames = b"\x00" * (n_frames * n_channels * sample_width)

            out_buf = io.BytesIO()
            with wave.open(out_buf, "wb") as out_wav:
                out_wav.setnchannels(n_channels)
                out_wav.setsampwidth(sample_width)
                out_wav.setframerate(framerate)
                out_wav.writeframes(silence_frames)

            return out_buf.getvalue()
        except Exception:
            return b""
