from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.schemas.tts import EdgeTTSRequest, VoiceInfo
from app.services.edge_engine import EdgeTTSEngine
from app.core.security import verify_token

router = APIRouter()

@router.get("/voices", response_model=list[VoiceInfo], dependencies=[Depends(verify_token)])
async def get_voices(engine: str = Query("edge", pattern="^(edge|qwen)$")):
    if engine == "edge":
        voices = await EdgeTTSEngine.get_voices()
        # Transform edge_tts voice dict to our schema
        return [
            VoiceInfo(
                id=v["Name"],
                name=v["FriendlyName"],
                engine="edge",
                gender=v.get("Gender"),
                locale=v.get("Locale")
            ) for v in voices
        ]
    
    # Placeholder for Qwen
    return []

class TTSRequest(BaseModel):
    text: str
    engine: str = "edge"
    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

@router.post("/stream", dependencies=[Depends(verify_token)])
async def tts_stream(request: TTSRequest):
    """
    Unified TTS streaming endpoint.
    Dispatches to the appropriate engine based on 'engine' field.
    """
    try:
        if request.engine == "edge":
            audio_generator = EdgeTTSEngine.generate_stream(
                request.text, 
                request.voice, 
                request.rate, 
                request.pitch
            )
        elif request.engine == "qwen":
            # Placeholder for Qwen implementation
            raise HTTPException(status_code=501, detail="Qwen engine not implemented yet")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported engine: {request.engine}")
            
        return StreamingResponse(audio_generator, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/edge/stream", dependencies=[Depends(verify_token)])
async def edge_tts_stream(request: EdgeTTSRequest):
    # Keep this for backward compatibility or direct access
    try:
        audio_generator = EdgeTTSEngine.generate_stream(
            request.text, 
            request.voice, 
            request.rate, 
            request.pitch
        )
        return StreamingResponse(audio_generator, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
