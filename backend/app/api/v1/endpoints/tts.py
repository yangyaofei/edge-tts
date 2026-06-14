from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse

from app.schemas.tts import VoiceInfo, TTSRequest
from app.core.security import verify_token
from app.core.config import settings
from app.services.registry import EngineRegistry, register_builtin_engines
from app.services.pipeline import TTSPipeline
from app.services.text_preprocessor import TextPreprocessor
from app.services.polyphone import PolyphoneFixer
from app.services.chunker import TextChunker
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
async def tts_stream(request: TTSRequest):
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
