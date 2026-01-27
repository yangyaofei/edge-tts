"""
TTS 数据模型定义

本模块定义了 TTS API 的请求和响应数据模型。
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# ===================================================================
# 文本处理 Schemas
# ===================================================================

class TextChunkRequest(BaseModel):
    """文本分块请求"""
    text: str = Field(..., description="要分块的文本")
    strategy: str = Field(default="paragraph", description="分块策略: paragraph, sentence, punctuation")

class Chunk(BaseModel):
    """文本块"""
    id: int = Field(..., description="块编号")
    text: str = Field(..., description="文本内容")

class TextChunkResponse(BaseModel):
    """文本分块响应"""
    chunks: List[Chunk] = Field(..., description="文本块列表")


# ===================================================================
# Edge TTS Schemas
# ===================================================================

class EdgeTTSRequest(BaseModel):
    """Edge TTS 请求模型"""
    text: str = Field(..., description="要转换的文本")
    voice: str = Field(default="zh-CN-XiaoxiaoNeural", description="语音 ID（如 zh-CN-XiaoxiaoNeural）")
    rate: str = Field(default="+0%", description="语速调整（如 +0%, -10%, +20%）")
    pitch: str = Field(default="+0Hz", description="音调调整（如 +0Hz, -10Hz, +10Hz）")


# ===================================================================
# 语音信息 Schemas
# ===================================================================

class VoiceInfo(BaseModel):
    """语音/说话人信息"""
    id: str = Field(..., description="语音/说话人 ID")
    name: str = Field(..., description="显示名称")
    engine: str = Field(..., description="TTS 引擎（edge 或 qwen_tts）")
    gender: Optional[str] = Field(None, description="性别（仅 Edge TTS）")
    locale: Optional[str] = Field(None, description="地区/语言")
    description: Optional[str] = Field(None, description="语音描述（仅 Qwen TTS）")


# ===================================================================
# Qwen3-TTS 相关类型定义
# ===================================================================

# 支持的语言
QwenLanguage = Literal[
    "Auto",      # 自动检测
    "Chinese",  # 中文
    "English",  # 英语
    "Japanese", # 日语
    "Korean",   # 韩语
    "French",   # 法语
    "German",   # 德语
    "Spanish",  # 西班牙语
    "Portuguese", # 葡萄牙语
    "Russian",  # 俄语
]

# 支持的说话人（Qwen3-TTS CustomVoice 模型）
QwenSpeaker = Literal[
    "Vivian",    # 明亮年轻女性（中文）
    "Serena",    # 温柔女性（中文）
    "Uncle_Fu",  # 成熟男性（中文）
    "Dylan",     # 北京腔男性（中文北京方言）
    "Eric",      # 成都腔男性（中文四川方言）
    "Ryan",      # 动感男性（英语）
    "Aiden",     # 阳光男性（英语）
    "Ono_Anna",  # 顽皮女性（日语）
    "Sohee",     # 温暖女性（韩语）
]


# ===================================================================
# 统一的 TTS 请求模型
# ===================================================================

class UnifiedTTSRequest(BaseModel):
    """
    统一的 TTS 请求模型

    支持 Edge TTS 和 Qwen3-TTS 两种引擎，通过 engine 参数选择。
    不同引擎支持不同的参数。
    """
    # 基础参数
    text: str = Field(
        ...,
        description="要转换的文本",
        min_length=1,
        max_length=5000
    )

    # 引擎选择
    engine: Literal["edge", "qwen_tts"] = Field(
        default="edge",
        description="TTS 引擎选择：edge 或 qwen_tts"
    )

    # Edge TTS 参数
    voice: Optional[str] = Field(
        default="zh-CN-XiaoxiaoNeural",
        description="语音 ID（Edge TTS: zh-CN-XiaoxiaoNeural，Qwen TTS: 说话人名称）"
    )
    rate: Optional[str] = Field(
        default="+0%",
        description="语速调整（仅 Edge TTS 有效，如 +0%, -10%, +20%）"
    )
    pitch: Optional[str] = Field(
        default="+0Hz",
        description="音调调整（仅 Edge TTS 有效，如 +0Hz, -10Hz, +10Hz）"
    )

    # Qwen TTS 参数
    speaker: QwenSpeaker = Field(
        default="Vivian",
        description="说话人（Qwen TTS：Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee）"
    )
    language: QwenLanguage = Field(
        default="Chinese",
        description="语言（Qwen TTS：Auto, Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian）"
    )
    instruct: Optional[str] = Field(
        default=None,
        description="风格指令（Qwen TTS 可选，如 '用温柔的声音说'）"
    )
