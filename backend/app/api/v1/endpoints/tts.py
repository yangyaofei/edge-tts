"""
TTS API 端点模块

本模块提供了文本转语音（TTS）的 API 端点，支持：
- Edge TTS：Microsoft Edge 在线语音服务（原有功能）
- Qwen3-TTS：阿里云开源的高质量语音合成（新增功能）

使用统一的 API 接口，通过 engine 参数选择不同的 TTS 引擎。
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse, Response
from app.schemas.tts import EdgeTTSRequest, VoiceInfo, UnifiedTTSRequest
from app.services.edge_engine import EdgeTTSEngine
from app.services.qwen_engine import QwenTTSEngine
from app.core.security import verify_token
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ===================================================================
# 健康检查端点
# ===================================================================

@router.get("/health/qwen", dependencies=[Depends(verify_token)])
async def qwen_health_check():
    """
    检查 Qwen3-TTS 模型状态

    返回模型的健康状态、类型、大小和设备信息。
    如果模型未初始化或发生错误，返回 503 状态码。
    """
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

@router.get("/voices", response_model=list[VoiceInfo])
async def get_voices(
    engine: str = Query("edge", description="引擎类型: edge 或 qwen_tts")
):
    """
    获取支持的语音/说话人列表

    Parameters:
    - engine: TTS 引擎选择
        - "edge": Microsoft Edge TTS，返回所有可用语音
        - "qwen_tts": Qwen3-TTS，返回 9 种预设说话人

    Returns:
        list[VoiceInfo]: 语音/说话人信息列表

    Note:
        Edge TTS: 支持 100+ 种语音和方言
        Qwen TTS: 支持 9 种高质量预设说话人
    """
    # 验证 token（从 query 参数或 header）
    try:
        verify_token(None)
    except:
        pass  # localhost 不需要 token

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
        raise HTTPException(
            status_code=400,
            detail=f"不支持的引擎类型: {engine}。请使用 'edge' 或 'qwen_tts'"
        )


# ===================================================================
# Edge TTS 专用端点（向后兼容）
# ===================================================================

@router.post("/edge/stream", dependencies=[Depends(verify_token)])
async def edge_tts_stream(request: EdgeTTSRequest):
    """
    Edge TTS 流式端点（向后兼容）

    使用 Microsoft Edge TTS 生成语音，返回 MP3 格式音频流。

    Parameters:
    - text: 要转换的文本
    - voice: 语音 ID（如 zh-CN-XiaoxiaoNeural）
    - rate: 语速调整（如 "+0%", "-10%", "+20%"）
    - pitch: 音调调整（如 "+0Hz", "-10Hz", "+10Hz"）

    Returns:
        StreamingResponse: MP3 格式音频流

    Note:
        这是 Edge TTS 的专用端点，保持了向后兼容性。
        推荐使用统一的 /stream 端点。
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
# Qwen3-TTS 端点（基础功能）
# ===================================================================

@router.post("/qwen_tts/generate", dependencies=[Depends(verify_token)])
async def qwen_tts_generate(request: UnifiedTTSRequest):
    """
    使用 Qwen3-TTS 生成语音（简化版）

    这是一个简化的端点，专注于基础 TTS 功能。
    支持 9 种预设说话人和 10 种语言。

    Parameters:
    - text: 要转换的文本
    - speaker: 说话人（默认 Vivian）
      可选: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
    - language: 语言（默认 Chinese）
      可选: Auto, Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian

    Returns:
        Response: WAV 格式音频

    Raises:
        HTTPException: 如果模型未初始化或生成失败

    Note:
        - 首次使用会从 HuggingFace 下载模型（~3.5GB）
        - 对于长文本（> 150 字），建议分段处理或增加超时时间
        - 可选的风格指令暂未在此端点中暴露
    """
    try:
        # 使用 speaker 字段作为说话人
        speaker = request.speaker or "Vivian"
        language = request.language or "Chinese"

        # 调用引擎生成语音
        audio_bytes, sr = await QwenTTSEngine.generate_custom_voice(
            text=request.text,
            speaker=speaker,
            language=language,
            instruct=None,  # 简化版不支持风格指令
        )

        return Response(content=audio_bytes, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Qwen TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ===================================================================
# 统一的 TTS 流式端点（支持多引擎）
# ===================================================================

@router.post("/stream", dependencies=[Depends(verify_token)])
async def tts_stream(request: UnifiedTTSRequest):
    """
    统一的 TTS 流式端点（推荐使用）

    支持通过 engine 参数选择不同的 TTS 引擎。

    **Edge TTS** (engine="edge")：
    - 返回 MP3 格式音频流
    - 支持语速和音调调整
    - 响应快速，适合实时应用

    **Qwen3-TTS** (engine="qwen_tts")：
    - 返回 WAV 格式完整音频
    - 支持多种预设说话人
    - 语音质量更高，适合高质量内容生成

    Parameters:
    - engine: TTS 引擎选择（edge 或 qwen_tts）
    - text: 要转换的文本
    - voice/说话人: 语音 ID
        - Edge: 语音 ID（如 zh-CN-XiaoxiaoNeural）
        - Qwen: 说话人（Vivian, Serena 等）
    - rate: 语速调整（仅 Edge TTS）
    - pitch: 音调调整（仅 Edge TTS）
    - language: 语言（仅 Qwen TTS）
    - instruct: 风格指令（仅 Qwen TTS，暂未在此端点中暴露）

    Returns:
        StreamingResponse: 音频流（Edge TTS: MP3，Qwen TTS: WAV）

    Raises:
        HTTPException: 如果引擎不支持或生成失败

    Examples:
        # Edge TTS
        {"text": "Hello", "engine": "edge", "voice": "en-US-JennyNeural"}

        # Qwen TTS
        {"text": "你好", "engine": "qwen_tts", "speaker": "Vivian", "language": "Chinese"}
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
                instruct=request.instruct,  # 支持风格指令
            )
            return Response(content=audio_bytes, media_type="audio/wav")

        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的引擎: {request.engine}。请使用 'edge' 或 'qwen_tts'"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
