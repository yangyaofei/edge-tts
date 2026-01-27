from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# --- Text Processing Schemas ---
class TextChunkRequest(BaseModel):
    text: str
    strategy: str = "paragraph" # paragraph, sentence, punctuation

class Chunk(BaseModel):
    id: int
    text: str

class TextChunkResponse(BaseModel):
    chunks: List[Chunk]

# --- TTS Schemas ---
class EdgeTTSRequest(BaseModel):
    text: str
    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

class VoiceInfo(BaseModel):
    id: str
    name: str
    engine: str
    gender: Optional[str] = None
    locale: Optional[str] = None
    description: Optional[str] = None

# --- Qwen3-TTS Schemas ---

# 支持的语言
QwenLanguage = Literal["Auto", "Chinese", "English", "Japanese", "Korean",
                       "French", "German", "Spanish", "Portuguese", "Russian"]

# 支持的说话人
QwenSpeaker = Literal["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
                      "Ryan", "Aiden", "Ono_Anna", "Sohee"]


class QwenTTSRequest(BaseModel):
    """Qwen3-TTS CustomVoice 模式请求"""
    text: str = Field(..., description="要转换的文本", min_length=1, max_length=5000)
    speaker: QwenSpeaker = Field(
        default="Vivian",
        description="说话人名称"
    )
    language: QwenLanguage = Field(
        default="Chinese",
        description="语言（Auto=自动检测）"
    )
    instruct: Optional[str] = Field(
        default=None,
        description="可选的风格指令，如：'用温柔的声音说'"
    )


class QwenVoiceDesignRequest(BaseModel):
    """Qwen3-TTS VoiceDesign 模式请求"""
    text: str = Field(..., description="要转换的文本", min_length=1, max_length=5000)
    instruct: str = Field(..., description="声音描述，如：'用温柔的声音说'")
    language: QwenLanguage = Field(
        default="Chinese",
        description="语言"
    )


class QwenVoiceCloneRequest(BaseModel):
    """Qwen3-TTS VoiceClone (Base模型) 请求"""
    text: str = Field(..., description="要转换的文本", min_length=1, max_length=5000)
    ref_audio_url: str = Field(..., description="参考音频的 URL 或文件路径")
    ref_text: str = Field(..., description="参考音频的文字内容")
    language: QwenLanguage = Field(
        default="Chinese",
        description="语言"
    )
    x_vector_only: bool = Field(
        default=False,
        description="是否仅使用 x-vector（不需要 ref_text，但质量会降低）"
    )


# 统一的 TTS 请求（支持多引擎）
class UnifiedTTSRequest(BaseModel):
    text: str = Field(..., description="要转换的文本")
    engine: Literal["edge", "qwen_tts"] = Field(
        default="edge",
        description="TTS 引擎选择"
    )
    # Edge TTS 参数
    voice: Optional[str] = Field(
        default=None,
        description="语音 ID（Edge: 如 zh-CN-XiaoxiaoNeural, Qwen: 如 Vivian）"
    )
    rate: Optional[str] = Field(
        default="+0%",
        description="语速调整（仅 Edge TTS 支持）"
    )
    pitch: Optional[str] = Field(
        default="+0Hz",
        description="音调调整（仅 Edge TTS 支持）"
    )
    # Qwen TTS 参数
    speaker: Optional[QwenSpeaker] = Field(
        default="Vivian",
        description="说话人（Qwen TTS）"
    )
    language: Optional[QwenLanguage] = Field(
        default="Chinese",
        description="语言（Qwen TTS）"
    )
    instruct: Optional[str] = Field(
        default=None,
        description="风格指令（Qwen TTS）"
    )
