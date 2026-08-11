from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import StreamingResponse

from app.schemas.tts import VoiceInfo, TTSRequest, SegmentRequest, SegmentSentence, SegmentResponse
from app.core.security import verify_token
from app.core.config import settings
from app.services.registry import EngineRegistry, register_builtin_engines
from app.services.pipeline import TTSPipeline
from app.services.text_preprocessor import TextPreprocessor
from app.services.polyphone import PolyphoneFixer
from app.services.chunker import TextChunker, split_sentences
from app.services.llm_transcriber import LLMTranscriber

logger = logging.getLogger(__name__)
router = APIRouter()

register_builtin_engines()


def _engine_kwargs(engine_name: str) -> dict:
    """根据 engine 名称返回构造参数。"""
    if engine_name == "qwen":
        return {
            "server_url": settings.QWEN3_TTS_SERVER_URL,
            "language": settings.QWEN3_TTS_LANGUAGE,
            "max_tokens": settings.QWEN3_TTS_MAX_TOKENS,
        }
    elif engine_name == "volcengine":
        return {
            "api_key": settings.VOLCENGINE_API_KEY,
            "app_id": settings.VOLCENGINE_APP_ID,
            "access_token": settings.VOLCENGINE_ACCESS_TOKEN,
        }
    return {}


def _build_pipeline(engine_name: str, preprocess: bool = True) -> TTSPipeline:
    """构建 pipeline 实例。"""
    engine = EngineRegistry.create(engine_name, **_engine_kwargs(engine_name))

    preprocessor = TextPreprocessor() if (preprocess and settings.TTS_PREPROCESS_ENABLED) else None
    polyphone_fixer = PolyphoneFixer() if (preprocess and settings.TTS_POLYPHONE_FIX_ENABLED) else None
    chunker = TextChunker()

    llm_transcriber = None
    if preprocess and settings.TTS_LLM_TRANSCRIBE_ENABLED:
        llm_transcriber = LLMTranscriber(
            api_url=settings.TTS_LLM_TRANSCRIBE_API_URL,
            api_key=settings.TTS_LLM_TRANSCRIBE_API_KEY,
            model=settings.TTS_LLM_TRANSCRIBE_MODEL,
        )

    ref_trim = settings.QWEN3_TTS_REF_TRIM_SECONDS if engine_name == "qwen" else 8

    return TTSPipeline(
        engine=engine,
        llm_transcriber=llm_transcriber,
        preprocessor=preprocessor,
        polyphone_fixer=polyphone_fixer,
        chunker=chunker,
        ref_trim_seconds=ref_trim,
        silence_between_chunks=settings.TTS_SILENCE_BETWEEN_CHUNKS,
        first_chunk_minimize=settings.TTS_FIRST_CHUNK_MINIMIZE,
    )


@router.get("/voices", response_model=list[VoiceInfo], dependencies=[Depends(verify_token)])
async def get_voices(engine: str = Query("edge", pattern="^(edge|qwen|volcengine)$")):
    try:
        engine_instance = EngineRegistry.create(engine, **_engine_kwargs(engine))
        voices = await engine_instance.get_voices()

        result = []
        for v in voices:
            result.append(VoiceInfo(
                id=v.get("id") or v.get("Name", ""),
                name=v.get("name") or v.get("FriendlyName", ""),
                engine=engine,
                gender=v.get("Gender") or v.get("gender"),
                locale=v.get("Locale") or v.get("locale"),
            ))
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stream", dependencies=[Depends(verify_token)])
async def tts_stream(request: TTSRequest, http_request: Request):
    """统一 TTS 流式端点。支持 edge / volcengine / qwen。"""
    if request.engine not in EngineRegistry.available():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported engine: {request.engine}. Available: {EngineRegistry.available()}",
        )

    try:
        pipeline = _build_pipeline(request.engine, request.preprocess)
        audio_gen = pipeline.generate_stream(
            request.text,
            voice=request.voice,
            speed=request.speed,
            use_preprocess=request.preprocess,
            request=http_request,
        )
        return StreamingResponse(audio_gen, media_type="audio/wav")
    except Exception as e:
        logger.error(f"TTS stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edge/stream", dependencies=[Depends(verify_token)])
async def edge_tts_stream(request: TTSRequest):
    """向后兼容端点。"""
    request.engine = "edge"
    return await tts_stream(request)


# ---------- /segment: 分句 + 归一化 (纯文本预处理, 不碰音频) ----------

# 模块级单例: 规则归一化无状态, 复用一个; LLM transcriber 按 endpoint 配置构建。
_PREPROCESSOR = TextPreprocessor()


def _build_segment_llm() -> LLMTranscriber | None:
    """构建 segment 专用的 LLM transcriber。

    与 _build_pipeline() 不同: 这里 raise_on_error=True —— 失败时抛异常
    而不是静默返回原文, 让 endpoint 能精确降级到 rule (source 可见)。
    """
    if not settings.TTS_LLM_TRANSCRIBE_ENABLED:
        return None
    if not (settings.TTS_LLM_TRANSCRIBE_API_URL and settings.TTS_LLM_TRANSCRIBE_API_KEY):
        return None
    return LLMTranscriber(
        api_url=settings.TTS_LLM_TRANSCRIBE_API_URL,
        api_key=settings.TTS_LLM_TRANSCRIBE_API_KEY,
        model=settings.TTS_LLM_TRANSCRIBE_MODEL,
        raise_on_error=True,
        timeout=settings.TTS_LLM_TIMEOUT,
    )


@router.post("/segment", response_model=SegmentResponse, dependencies=[Depends(verify_token)])
async def tts_segment(request: SegmentRequest):
    """分句 + 归一化。返回逐句 original/tts_text, original 拼接 == 输入原文。

    normalize:
      - none: 不归一化, tts_text == original
      - rule: TextPreprocessor (本地正则)
      - llm:  LLMTranscriber 逐句并发, 单句失败降级 rule
    """
    text = (request.text or "").strip()
    if not text:
        return SegmentResponse(sentences=[])

    mode = request.normalize
    if mode not in ("none", "rule", "llm"):
        raise HTTPException(status_code=400, detail=f"normalize must be none|rule|llm, got {mode!r}")

    originals = split_sentences(text, min_len=4)
    if not originals:
        return SegmentResponse(sentences=[])

    llm = _build_segment_llm() if mode == "llm" else None
    sem = asyncio.Semaphore(settings.TTS_LLM_CONCURRENCY)

    async def normalize_one(sentence: str) -> tuple[str, str]:
        """单句归一化, 返回 (tts_text, source)。"""
        if mode == "none":
            return sentence, "none"
        if mode == "rule" or llm is None:
            return _PREPROCESSOR.process(sentence), "rule"
        # mode == "llm"
        try:
            async with sem:
                tts_text = await asyncio.wait_for(llm.transcribe(sentence), timeout=settings.TTS_LLM_TIMEOUT)
            tts_text = (tts_text or "").strip()
            if not tts_text:
                raise ValueError("LLM returned empty text")
            return tts_text, "llm"
        except Exception as e:
            logger.warning(f"segment LLM normalize failed, fallback to rule: {e}")
            return _PREPROCESSOR.process(sentence), "rule"

    results = await asyncio.gather(*(normalize_one(s) for s in originals))

    sentences = [
        SegmentSentence(index=i, original=orig, tts_text=tts, source=src)
        for i, (orig, (tts, src)) in enumerate(zip(originals, results))
    ]
    return SegmentResponse(sentences=sentences)
