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


class _WavPcmExtractor:
    """从 WAV 字节流中剥离 header，只输出纯 PCM。

    用于多段拼接场景：qwen engine 每次返回完整 WAV（含 RIFF header），
    若直接拼接，中间的 header 字节会被当作 PCM 解码产生爆音。
    feed() 多次喂入字节：首次解析出完整 header 时返回 (fmt, 剩余PCM)，
    之后返回 (None, data) 透传。若输入不是 WAV（如 MP3），原样透传。
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._parsed = False
        self.fmt: dict | None = None

    def feed(self, data: bytes) -> tuple[dict | None, bytes]:
        if self._parsed:
            return None, data
        self._buf.extend(data)
        parsed = self._try_parse()
        if parsed is None:
            return None, b""
        fmt, pcm_offset = parsed
        self.fmt = fmt
        self._parsed = True
        remaining = bytes(self._buf[pcm_offset:])
        self._buf = bytearray()
        return fmt, remaining

    def _try_parse(self) -> tuple[dict, int] | None:
        buf = self._buf
        if len(buf) < 12:
            return None
        if buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
            return ({}, 0)
        fmt: dict = {}
        pos = 12
        while pos + 8 <= len(buf):
            chunk_id = bytes(buf[pos:pos + 4])
            chunk_size = struct.unpack("<I", bytes(buf[pos + 4:pos + 8]))[0]
            body_start = pos + 8
            if chunk_id == b"fmt ":
                if body_start + 16 > len(buf):
                    return None
                _af, channels, sample_rate, _br, _ba, bits = struct.unpack(
                    "<HHIIHH", bytes(buf[body_start:body_start + 16])
                )
                fmt = {
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "sample_width": bits // 8,
                }
            elif chunk_id == b"data":
                return (fmt, body_start)
            pos = body_start + chunk_size
            if chunk_size & 1:
                pos += 1
        return None


def _build_streaming_wav_header(fmt: dict) -> bytes:
    """构造流式 WAV header（data size = 0x7FFFFFFF，表示未知长度）。"""
    sr = fmt.get("sample_rate", 24000)
    channels = fmt.get("channels", 1)
    sample_width = fmt.get("sample_width", 2)
    bits = sample_width * 8
    byte_rate = sr * channels * bits // 8
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0x7FFFFFFF, b"WAVE",
        b"fmt ", 16, 1, channels, sr, byte_rate, block_align, bits,
        b"data", 0x7FFFFFFF,
    )


def _make_silence_pcm(fmt: dict, seconds: float) -> bytes:
    """生成纯 PCM 静音（无 WAV header）。"""
    sr = fmt.get("sample_rate", 24000)
    channels = fmt.get("channels", 1)
    sample_width = fmt.get("sample_width", 2)
    n = int(seconds * sr) * channels * sample_width
    return b"\x00" * n


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
        request=None,
        **engine_kwargs,
    ) -> AsyncGenerator[bytes, None]:
        """完整 pipeline 流式生成。

        Yields: WAV bytes
        request: Starlette Request 对象，用于检测客户端断开
        engine_kwargs: 传递给 engine 的额外参数 (temperature, instruct, etc.)
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
        # 整个输出流只含一个 WAV header（开头），后续所有段和静音都是纯 PCM。
        # 避免每段 RIFF header 被当作 PCM 解码产生爆音（"RIFF"=0x52494646 → 大幅值样本）。
        ref_audio: bytes | None = None
        wav_fmt: dict | None = None
        header_sent = False

        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            # 客户端断开检测（httpx cancel scope 会吞掉 CancelledError，
            # 导致 pipeline 误以为当前段正常结束，继续发新请求。
            # 必须主动检测断开并停止。）
            if request is not None and await request.is_disconnected():
                logger.info(f"Pipeline: client disconnected, stopping at chunk {i}/{len(chunks)}")
                return

            logger.debug(f"Pipeline chunk {i}/{len(chunks)}: {len(chunk_text)} chars")

            if hasattr(self.engine, 'generate_chunk_stream'):
                # 流式 engine（qwen）：每次返回完整 WAV，需剥离 header
                extractor = _WavPcmExtractor()
                need_ref = i == 0 and len(chunks) > 1
                chunk_pcm_parts: list[bytes] = [] if need_ref else None

                # 段间静音（纯 PCM，用首段解析出的格式）
                if i > 0 and self.silence_between_chunks > 0 and wav_fmt is not None:
                    yield _make_silence_pcm(wav_fmt, self.silence_between_chunks)

                async for data in self.engine.generate_chunk_stream(
                    chunk_text, voice=voice, speed=speed, ref_audio=ref_audio,
                    **engine_kwargs,
                ):
                    fmt, pcm = extractor.feed(data)
                    # 首次解析出 WAV header → 发送流式 header（仅一次）
                    if fmt is not None and not header_sent:
                        wav_fmt = fmt or {
                            "sample_rate": self.sample_rate,
                            "channels": 1,
                            "sample_width": 2,
                        }
                        yield _build_streaming_wav_header(wav_fmt)
                        header_sent = True
                    if pcm:
                        yield pcm
                        if chunk_pcm_parts is not None:
                            chunk_pcm_parts.append(pcm)

                # 用首段 PCM 构造 ref_audio（声音克隆一致性，engine 可选支持）
                if need_ref and chunk_pcm_parts and wav_fmt:
                    ref_audio = self._build_wav_from_pcm(
                        b"".join(chunk_pcm_parts), wav_fmt,
                    )
            else:
                # 非流式 engine（edge/volcengine，返回 MP3）：拼接 MP3 帧无爆音
                audio = await self.engine.generate_chunk(
                    chunk_text, voice=voice, speed=speed, ref_audio=ref_audio,
                )
                if i == 0 and len(chunks) > 1:
                    ref_audio = self._extract_ref(audio, self.ref_trim_seconds)
                if i > 0 and self.silence_between_chunks > 0:
                    yield self._make_silence(audio, self.silence_between_chunks)
                else:
                    yield audio

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

    def _make_silence_from_engine(self, ref_wav: bytes | None, seconds: float) -> bytes:
        """从引擎格式信息生成静音 PCM（用于流式模式）。"""
        if ref_wav:
            return self._make_silence(ref_wav, seconds)
        # 默认格式：16-bit PCM mono 24kHz
        n_samples = int(seconds * self.sample_rate)
        return b"\x00" * (n_samples * 2)  # int16 = 2 bytes/sample

    def _build_wav_from_pcm(self, pcm: bytes, fmt: dict) -> bytes:
        """用纯 PCM + 格式信息构造完整 WAV bytes（用于 ref_audio）。"""
        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as out_wav:
            out_wav.setnchannels(fmt.get("channels", 1))
            out_wav.setsampwidth(fmt.get("sample_width", 2))
            out_wav.setframerate(fmt.get("sample_rate", self.sample_rate))
            out_wav.writeframes(pcm)
        return out_buf.getvalue()
