from __future__ import annotations

import base64
import json
import logging

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import Response, StreamingResponse

from app.schemas.tts import OPENAI_AUDIO_CONTENT_TYPES
from app.services.registry import EngineRegistry, register_builtin_engines
from app.services.pipeline import TTSPipeline
from app.services.text_preprocessor import TextPreprocessor
from app.services.polyphone import PolyphoneFixer
from app.services.chunker import TextChunker
from app.services.llm_transcriber import LLMTranscriber
from app.core.security import verify_token
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

register_builtin_engines()

VOICE_TO_ENGINE: dict[str, str] = {
    "alloy": "volcengine", "echo": "volcengine",
    "shimmer": "volcengine", "fable": "volcengine",
    "nova": "volcengine", "onyx": "volcengine",
    "coral": "volcengine", "sage": "volcengine",
}


def _resolve_engine_and_voice(model: str, voice_raw) -> tuple[str, str]:
    if model == "volcengine":
        voice_id = voice_raw.get("id", "alloy") if isinstance(voice_raw, dict) else voice_raw
        return "volcengine", voice_id
    if model == "qwen":
        return "qwen", "default"
    if isinstance(voice_raw, str) and voice_raw in VOICE_TO_ENGINE:
        return VOICE_TO_ENGINE[voice_raw], voice_raw
    voice_id = voice_raw.get("id", "zh-CN-XiaoxiaoNeural") if isinstance(voice_raw, dict) else voice_raw
    return "edge", voice_id


def _engine_kwargs(engine_name: str) -> dict:
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


def _build_pipeline(engine_name: str) -> TTSPipeline:
    engine = EngineRegistry.create(engine_name, **_engine_kwargs(engine_name))

    llm_transcriber = None
    if settings.TTS_LLM_TRANSCRIBE_ENABLED:
        llm_transcriber = LLMTranscriber(
            api_url=settings.TTS_LLM_TRANSCRIBE_API_URL,
            api_key=settings.TTS_LLM_TRANSCRIBE_API_KEY,
            model=settings.TTS_LLM_TRANSCRIBE_MODEL,
        )

    return TTSPipeline(
        engine=engine,
        llm_transcriber=llm_transcriber,
        preprocessor=TextPreprocessor() if settings.TTS_PREPROCESS_ENABLED else None,
        polyphone_fixer=PolyphoneFixer() if settings.TTS_POLYPHONE_FIX_ENABLED else None,
        chunker=TextChunker(),
        ref_trim_seconds=settings.QWEN3_TTS_REF_TRIM_SECONDS,
        silence_between_chunks=settings.TTS_SILENCE_BETWEEN_CHUNKS,
        first_chunk_minimize=settings.TTS_FIRST_CHUNK_MINIMIZE,
    )


@router.post("/v1/audio/speech", dependencies=[Depends(verify_token)])
async def create_speech(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": {"type": "invalid_request_error", "message": "invalid JSON"}})

    text = body.get("input", "")
    if not text:
        raise HTTPException(status_code=400, detail={"error": {"type": "invalid_request_error", "message": "input is required", "param": "input"}})

    model = body.get("model", "tts-1")
    voice_raw = body.get("voice", "alloy")
    audio_format = body.get("response_format", "mp3") or "mp3"
    speed = body.get("speed", 1.0) or 1.0
    stream_format = body.get("stream_format", "audio") or "audio"
    instructions = body.get("instructions")  # OpenAI's natural language instruction
    temperature = body.get("temperature")    # Custom extension: control randomness

    engine_type, voice_id = _resolve_engine_and_voice(model, voice_raw)

    logger.info(f"TTS: engine={engine_type} voice={voice_id} format={audio_format} speed={speed} text_len={len(text)}")

    content_type = OPENAI_AUDIO_CONTENT_TYPES.get(audio_format, "application/octet-stream")

    try:
        pipeline = _build_pipeline(engine_type)

        # Pass engine-specific params
        extra_kwargs = {}
        if engine_type == "qwen":
            if instructions:
                extra_kwargs["instruct"] = instructions
            if temperature is not None:
                extra_kwargs["temperature"] = temperature

        audio_gen = pipeline.generate_stream(
            text, voice=voice_id, speed=speed,
            **extra_kwargs,
        )

        if stream_format == "sse":
            async def sse_stream():
                async for chunk in audio_gen:
                    event = {"type": "speech.audio.delta", "audio": base64.b64encode(chunk).decode()}
                    yield f"data: {json.dumps(event)}\n\n"
                done = {"type": "speech.audio.done", "usage": {"input_tokens": len(text), "output_tokens": 0, "total_tokens": len(text)}}
                yield f"data: {json.dumps(done)}\n\n"
            return StreamingResponse(sse_stream(), media_type="text/event-stream")

        return StreamingResponse(audio_gen, media_type=content_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail={"error": {"type": "server_error", "message": str(e)}})


@router.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model", "owned_by": "edge-tts"},
            {"id": "tts-1-hd", "object": "model", "owned_by": "edge-tts"},
            {"id": "gpt-4o-mini-tts", "object": "model", "owned_by": "edge-tts"},
            {"id": "volcengine", "object": "model", "owned_by": "volcengine"},
            {"id": "qwen", "object": "model", "owned_by": "qwen3-tts"},
        ],
    }


@router.get("/v1/audio/voices", dependencies=[Depends(verify_token)])
async def list_voices():
    all_voices = []
    for engine_name in EngineRegistry.available():
        try:
            engine = EngineRegistry.create(engine_name, **_engine_kwargs(engine_name))
            voices = await engine.get_voices()
            for v in voices:
                v["engine"] = engine_name
                all_voices.append(v)
        except Exception as e:
            logger.warning(f"Failed to get voices for {engine_name}: {e}")
    return {"object": "list", "data": all_voices}
