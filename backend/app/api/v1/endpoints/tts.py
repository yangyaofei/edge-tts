from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, Response
from app.schemas.tts import (
    EdgeTTSRequest, VoiceInfo,
    QwenTTSRequest, QwenVoiceDesignRequest, QwenVoiceCloneRequest,
    UnifiedTTSRequest
)
from app.services.edge_engine import EdgeTTSEngine
from app.services.qwen_engine import QwenTTSEngine
from app.core.security import verify_token
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ===================================================================
# 健康检查和模型信息
# ===================================================================

@router.get("/health/qwen", dependencies=[Depends(verify_token)])
async def qwen_health_check():
    """检查 Qwen3-TTS 模型状态"""
    try:
        health = await QwenTTSEngine.health_check()
        status_code = 200 if health["status"] == "healthy" else 503
        return health
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "model": QwenTTSEngine.get_model_info()
        }


# ===================================================================
# 获取语音列表
# ===================================================================

@router.get("/voices", response_model=list[VoiceInfo], dependencies=[Depends(verify_token)])
async def get_voices(engine: str = Query("edge", pattern="^(edge|qwen_tts)$")):
    """
    获取支持的语音列表

    参数:
    - engine: 引擎类型 (edge, qwen_tts)
    """
    if engine == "edge":
        voices = await EdgeTTSEngine.get_voices()
        return [
            VoiceInfo(
                id=v["Name"],
                name=v["FriendlyName"],
                engine="edge",
                gender=v.get("Gender"),
                locale=v.get("Locale")
            ) for v in voices
        ]
    elif engine == "qwen_tts":
        # 返回 Qwen 预设说话人
        speakers_data = {
            "Vivian": ("明亮、略带棱角的年轻女性声音", "中文"),
            "Serena": ("温柔、温和的年轻女性声音", "中文"),
            "Uncle_Fu": ("成熟、低沉醇厚的男声", "中文"),
            "Dylan": ("清晰自然的北京腔年轻男声", "中文（北京方言）"),
            "Eric": ("活泼、略带沙哑明亮的成都男声", "中文（四川方言）"),
            "Ryan": ("动感十足、节奏感强的男声", "英语"),
            "Aiden": ("阳光开朗的美国男声", "英语"),
            "Ono_Anna": ("顽皮轻盈的日本女性声音", "日语"),
            "Sohee": ("温暖情感丰富的韩国女性声音", "韩语"),
        }
        return [
            VoiceInfo(
                id=speaker,
                name=speaker,
                engine="qwen_tts",
                description=desc,
                locale=lang
            )
            for speaker, (desc, lang) in speakers_data.items()
        ]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported engine: {engine}")


# ===================================================================
# Qwen3-TTS CustomVoice 模式端点
# ===================================================================

@router.post("/qwen_tts/generate", dependencies=[Depends(verify_token)])
async def qwen_tts_generate(request: QwenTTSRequest):
    """
    使用 Qwen3-TTS CustomVoice 模式生成语音

    参数:
    - text: 要转换的文本
    - speaker: 说话人 (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee)
    - language: 语言 (Auto, Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian)
    - instruct: 可选的风格指令，如 "用温柔的声音说"

    返回: WAV 格式音频
    """
    try:
        audio_bytes, sr = await QwenTTSEngine.generate_custom_voice(
            text=request.text,
            speaker=request.speaker,
            language=request.language,
            instruct=request.instruct,
        )
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error(f"Qwen TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qwen_tts/design", dependencies=[Depends(verify_token)])
async def qwen_tts_design(request: QwenVoiceDesignRequest):
    """
    使用 Qwen3-TTS VoiceDesign 模式生成语音（自然语言描述声音）

    参数:
    - text: 要转换的文本
    - instruct: 声音描述，如 "用温柔的声音说"
    - language: 语言

    返回: WAV 格式音频
    """
    try:
        audio_bytes, sr = await QwenTTSEngine.generate_voice_design(
            text=request.text,
            instruct=request.instruct,
            language=request.language,
        )
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error(f"Qwen VoiceDesign error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qwen_tts/clone", dependencies=[Depends(verify_token)])
async def qwen_tts_clone(request: QwenVoiceCloneRequest):
    """
    使用 Qwen3-TTS Base 模型进行声音克隆

    参数:
    - text: 要转换的文本
    - ref_audio_url: 参考音频的 URL 或文件路径
    - ref_text: 参考音频的文字内容
    - language: 语言
    - x_vector_only: 是否仅使用 x-vector（不需要 ref_text，但质量会降低）

    返回: WAV 格式音频
    """
    try:
        audio_bytes, sr = await QwenTTSEngine.generate_voice_clone(
            text=request.text,
            ref_audio_path=request.ref_audio_url,
            ref_text=request.ref_text,
            language=request.language,
            x_vector_only=request.x_vector_only,
        )
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error(f"Qwen VoiceClone error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ===================================================================
# Edge TTS 端点（保持向后兼容）
# ===================================================================

@router.post("/edge/stream", dependencies=[Depends(verify_token)])
async def edge_tts_stream(request: EdgeTTSRequest):
    """
    Edge TTS 流式端点（向后兼容）

    返回: MP3 格式音频流
    """
    try:
        audio_generator = EdgeTTSEngine.generate_stream(
            request.text,
            request.voice,
            request.rate,
            request.pitch
        )
        return StreamingResponse(audio_generator, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"Edge TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ===================================================================
# 统一的 TTS 流式端点（支持多引擎）
# ===================================================================

@router.post("/stream", dependencies=[Depends(verify_token)])
async def tts_stream(request: UnifiedTTSRequest):
    """
    统一的 TTS 流式端点

    支持 engine 参数选择不同的 TTS 引擎：
    - edge: Microsoft Edge TTS (返回 MP3 流)
    - qwen_tts: Qwen3-TTS (返回 WAV 完整音频)

    参数说明:
    - engine: 引擎选择 (edge, qwen_tts)
    - text: 要转换的文本
    - voice: 语音 ID (Edge: zh-CN-XiaoxiaoNeural, Qwen: Vivian)
    - speaker: 说话人 (Qwen TTS，默认 Vivian)
    - language: 语言 (Qwen TTS，默认 Chinese)
    - rate/pitch: 语速/音调 (仅 Edge TTS)
    - instruct: 风格指令 (Qwen TTS，可选)
    """
    try:
        if request.engine == "edge":
            # Edge TTS 流式输出
            voice = request.voice or "zh-CN-XiaoxiaoNeural"
            audio_generator = EdgeTTSEngine.generate_stream(
                request.text,
                voice,
                request.rate or "+0%",
                request.pitch or "+0Hz"
            )
            return StreamingResponse(audio_generator, media_type="audio/mpeg")

        elif request.engine == "qwen_tts":
            # Qwen TTS 返回完整音频
            speaker = request.speaker or "Vivian"
            language = request.language or "Chinese"

            audio_bytes, sr = await QwenTTSEngine.generate_custom_voice(
                text=request.text,
                speaker=speaker,
                language=language,
                instruct=request.instruct,
            )
            return Response(content=audio_bytes, media_type="audio/wav")

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported engine: {request.engine}. Use 'edge' or 'qwen_tts'"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
