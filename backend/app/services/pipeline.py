from __future__ import annotations

import logging
import re
import struct
import wave
import io
from typing import AsyncGenerator

from app.services.base import TTSEngine
from app.services.text_preprocessor import TextPreprocessor
from app.services.polyphone import PolyphoneFixer
from app.services.chunker import TextChunker
from app.services.llm_transcriber import LLMTranscriber

logger = logging.getLogger(__name__)


class TTSPipeline:
    """TTS 编排层：LLM转写 → 预处理 → 多音字 → 分段 → engine 合成。

    所有步骤都是可选的（传 None 跳过）。
    Engine 只负责 chunk 文本 → 音频 bytes。
    Pipeline 负责 ref_audio 状态管理、段间静音、首段最小化。
    """

    SENTENCE_SPLIT = re.compile(r"[。！？!？\.\n]")

    def __init__(
        self,
        engine: TTSEngine,
        llm_transcriber: LLMTranscriber | None = None,
        preprocessor: TextPreprocessor | None = None,
        polyphone_fixer: PolyphoneFixer | None = None,
        chunker: TextChunker | None = None,
        ref_trim_seconds: int = 8,
        silence_between_chunks: float = 0.3,
        first_chunk_minimize: bool = True,
        sample_rate: int = 24000,
    ) -> None:
        self.engine = engine
        self.llm_transcriber = llm_transcriber
        self.preprocessor = preprocessor
        self.polyphone_fixer = polyphone_fixer
        self.chunker = chunker
        self.ref_trim_seconds = ref_trim_seconds
        self.silence_between_chunks = silence_between_chunks
        self.first_chunk_minimize = first_chunk_minimize
        self.sample_rate = sample_rate

    async def generate_stream(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        use_preprocess: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        """完整 pipeline 流式生成。

        Yields: WAV bytes
        """
        processed = text

        # 0. LLM 转写（可选，最耗时的前置步骤）
        if use_preprocess and self.llm_transcriber and self.llm_transcriber.is_configured():
            processed = await self.llm_transcriber.transcribe(processed)

        # 1. 正则预处理
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

        # 3. 首段最小化：将第一段拆出第一句话，降低首字延迟
        if self.first_chunk_minimize and len(chunks) > 0 and len(chunks[0]) > 100:
            chunks = self._split_first_sentence(chunks)

        # 4. 逐段生成
        ref_audio: bytes | None = None
        first_chunk = True

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

            # 首段提取参考音频（用于后续段声音克隆）
            if first_chunk and len(chunks) > 1:
                ref_audio = self._extract_ref(audio, self.ref_trim_seconds)

            # 段间插入静音
            if not first_chunk and self.silence_between_chunks > 0:
                yield self._make_silence(audio, self.silence_between_chunks)
            else:
                yield audio

            first_chunk = False

    def _split_first_sentence(self, chunks: list[str]) -> list[str]:
        """将第一段拆为 [第一句, 剩余部分, ...其他段]，降低首字延迟。"""
        first = chunks[0]
        parts = self.SENTENCE_SPLIT.split(first, maxsplit=1)
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            return [parts[0].strip(), parts[1].strip()] + chunks[1:]
        return chunks

    def _extract_ref(self, wav_bytes: bytes, seconds: int) -> bytes:
        """从 WAV bytes 中截取前 N 秒作为参考音频。"""
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
            logger.warning(f"Failed to extract ref audio: {e}")
            return wav_bytes

    def _make_silence(self, ref_wav: bytes, seconds: float) -> bytes:
        """构造一段静音 WAV。"""
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
