import base64
import json
import logging

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import Response, StreamingResponse

from app.schemas.tts import OpenAITTSRequest, OPENAI_AUDIO_CONTENT_TYPES
from app.services.edge_engine import EdgeTTSEngine
from app.services.volcengine_engine import VolcengineTTSEngine
from app.core.security import verify_token
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_volcengine_engine() -> VolcengineTTSEngine:
    return VolcengineTTSEngine(
        api_key=settings.VOLCENGINE_API_KEY or "",
        app_id=settings.VOLCENGINE_APP_ID or "",
        access_token=settings.VOLCENGINE_ACCESS_TOKEN or "",
    )


def _resolve_voice(voice_raw, model: str) -> tuple[str, str]:
    if model == "volcengine":
        voice_id = voice_raw.get("id", "alloy") if isinstance(voice_raw, dict) else voice_raw
        return "volcengine", VolcengineTTSEngine.resolve_voice(voice_id)
    voice_id = voice_raw.get("id", "zh-CN-XiaoxiaoNeural") if isinstance(voice_raw, dict) else voice_raw
    return "edge", voice_id


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

    engine_type, voice_id = _resolve_voice(voice_raw, model)

    logger.info(f"TTS: engine={engine_type} voice={voice_id} format={audio_format} speed={speed} text_len={len(text)}")

    content_type = OPENAI_AUDIO_CONTENT_TYPES.get(audio_format, "application/octet-stream")

    try:
        if engine_type == "volcengine":
            volcengine = _get_volcengine_engine()
            audio_gen = volcengine.generate_stream(text, voice_id, speed, audio_format)
        else:
            rate_str = f"+{int((speed - 1.0) * 100)}%" if speed >= 1.0 else f"{int((speed - 1.0) * 100)}%"
            audio_gen = EdgeTTSEngine.generate_stream(text, voice_id, rate=rate_str, pitch="+0Hz")

        if stream_format == "sse":
            async def sse_stream():
                import json as _json
                async for chunk in audio_gen:
                    event = {"type": "speech.audio.delta", "audio": base64.b64encode(chunk).decode()}
                    yield f"data: {_json.dumps(event)}\n\n"
                done = {"type": "speech.audio.done", "usage": {"input_tokens": len(text), "output_tokens": 0, "total_tokens": len(text)}}
                yield f"data: {_json.dumps(done)}\n\n"
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
        ],
    }


@router.get("/v1/audio/voices", dependencies=[Depends(verify_token)])
async def list_voices():
    edge_voices_raw = await EdgeTTSEngine.get_voices()
    edge_voices = [
        {"id": v["Name"], "name": v["FriendlyName"], "engine": "edge", "locale": v.get("Locale")}
        for v in edge_voices_raw
    ]
    volcengine_voices = VolcengineTTSEngine.get_openai_voices()
    return {"object": "list", "data": edge_voices + volcengine_voices}
