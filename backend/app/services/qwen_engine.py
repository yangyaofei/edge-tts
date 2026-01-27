"""
Qwen3-TTS 引擎模块

本模块提供了 Qwen3-TTS 文本转语音模型的集成封装，支持：
- 多种设备后端：CUDA (NVIDIA GPU)、MPS (Apple Silicon)、CPU
- 多种模型类型：CustomVoice（预设音色）、Base（声音克隆）、VoiceDesign（声音设计）
- 自动设备检测和最优数据类型选择
- 线程安全的模型单例管理

作者：Claude AI
创建时间：2026-01-26
"""

import torch
import logging
import io
import asyncio
from typing import AsyncGenerator, Optional, Literal
from functools import lru_cache
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# ============================================================================
# 类型定义
# ============================================================================

# 支持的模型类型
ModelType = Literal["CustomVoice", "Base", "VoiceDesign"]
# - CustomVoice: 9 种预设高质量说话人，支持指令控制
# - Base: 支持 3 秒音频快速克隆声音
# - VoiceDesign: 通过自然语言描述设计自定义声音

# 支持的模型大小
ModelSize = Literal["0.6B", "1.7B"]
# - 0.6B: 轻量级模型，适合资源受限环境
# - 1.7B: 高质量模型，推荐用于生产环境

# 支持的语言
Language = Literal["Auto", "Chinese", "English", "Japanese", "Korean",
                   "French", "German", "Spanish", "Portuguese", "Russian"]

# CustomVoice 模型支持的预设说话人列表
CUSTOM_VOICE_SPEAKERS = [
    "Vivian",      # 明亮、略带棱角的年轻女性声音 (中文)
    "Serena",      # 温柔、温和的年轻女性声音 (中文)
    "Uncle_Fu",    # 成熟、低沉醇厚的男声 (中文)
    "Dylan",       # 清晰自然的北京腔年轻男声 (中文北京方言)
    "Eric",        # 活泼、略带沙哑明亮的成都男声 (中文四川方言)
    "Ryan",        # 动感十足、节奏感强的男声 (英语)
    "Aiden",       # 阳光开朗的美国男声 (英语)
    "Ono_Anna",    # 顽皮轻盈的日本女性声音 (日语)
    "Sohee",       # 温暖情感丰富的韩国女性声音 (韩语)
]

# 说话人描述映射（用于文档和 API 响应）
SPEAKER_DESCRIPTIONS = {
    "Vivian": "Bright, slightly edgy young female voice",
    "Serena": "Warm, gentle young female voice",
    "Uncle_Fu": "Seasoned male voice with a low, mellow timbre",
    "Dylan": "Youthful Beijing male voice with a clear, natural timbre",
    "Eric": "Lively Chengdu male voice with a slightly husky brightness",
    "Ryan": "Dynamic male voice with strong rhythmic drive",
    "Aiden": "Sunny American male voice with a clear midrange",
    "Ono_Anna": "Playful Japanese female voice with a light, nimble timbre",
    "Sohee": "Warm Korean female voice with rich emotion",
}


class QwenTTSEngine:
    """
    Qwen3-TTS 引擎类

    这是一个单例类，管理 Qwen3-TTS 模型的加载、初始化和推理。
    支持自动设备检测（CUDA > MPS > CPU）和最优数据类型选择。

    特性：
    - 线程安全的模型加载和访问
    - 自动设备检测和优化
    - 支持三种模式：CustomVoice、Base、VoiceDesign
    - 统一的语音生成接口

    使用示例：
        >>> await QwenTTSEngine.initialize(model_type="CustomVoice", model_size="1.7B")
        >>> audio_bytes, sr = await QwenTTSEngine.generate_custom_voice("你好", "Vivian")
    """

    # 类变量（单例模式）
    _model: Optional["Qwen3TTSModel"] = None  # 已加载的模型实例
    _model_type: Optional[ModelType] = None    # 当前模型类型
    _model_size: Optional[ModelSize] = None    # 当前模型大小
    _device: Optional[str] = None             # 当前使用的设备
    _lock: asyncio.Lock = asyncio.Lock()      # 异步锁，保证线程安全

    @classmethod
    def get_available_device(cls) -> str:
        """
        自动检测并返回可用的最佳设备

        设备优先级：CUDA > MPS > CPU

        Returns:
            str: 设备标识符 ("cuda:0", "mps", 或 "cpu")

        Note:
            - CUDA: NVIDIA GPU，使用 bfloat16 获得最佳性能
            - MPS: Apple Silicon GPU (M1/M2/M3/M4)，使用 float16
            - CPU: 降级选项，速度较慢但无硬件要求
        """
        if torch.cuda.is_available():
            device = "cuda:0"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return device
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS (Apple Silicon) device")
            return device
        else:
            logger.warning("No GPU detected, using CPU (inference will be slow)")
            return "cpu"

    @classmethod
    def get_optimal_dtype(cls, device: str) -> torch.dtype:
        """
        根据设备选择最优的数据类型

        Args:
            device: 设备标识符

        Returns:
            torch.dtype: 推荐的数据类型

        Note:
            - MPS: 使用 float16（更稳定，bfloat16 支持不完善）
            - CUDA: 使用 bfloat16（更好性能和精度）
            - CPU: 使用 float32（稳定性）
        """
        if device == "mps":
            # MPS 对 bfloat16 支持不完善，使用 float16
            logger.info("Using float16 for MPS device")
            return torch.float16
        elif device.startswith("cuda"):
            # CUDA 支持 bfloat16，性能更好
            logger.info("Using bfloat16 for CUDA device")
            return torch.bfloat16
        else:
            # CPU 使用 float32
            logger.info("Using float32 for CPU")
            return torch.float32

    @classmethod
    async def initialize(
        cls,
        model_type: ModelType = "CustomVoice",
        model_size: ModelSize = "1.7B",
        device: Optional[str] = None,
    ):
        """
        初始化 Qwen3-TTS 模型（单例模式）

        这是模型加载的入口点，会自动检测设备、选择最优数据类型，
        并加载模型到内存。使用单例模式确保全局只有一个模型实例。

        Args:
            model_type: 模型类型
                - "CustomVoice": 预设 9 种说话人，支持指令控制（推荐）
                - "Base": 声音克隆模型，需要参考音频
                - "VoiceDesign": 自然语言声音设计
            model_size: 模型大小
                - "1.7B": 高质量模型，推荐用于生产环境（默认）
                - "0.6B": 轻量级模型，适合资源受限环境
            device: 设备标识符（None=自动检测）
                - None: 自动检测 CUDA > MPS > CPU
                - "cuda:0": 强制使用第一块 NVIDIA GPU
                - "mps": 强制使用 Apple Silicon GPU
                - "cpu": 强制使用 CPU（速度慢）

        Raises:
            RuntimeError: 如果 qwen-tts 包未安装
            Exception: 如果模型加载失败

        Note:
            - 首次调用会从 HuggingFace 下载模型（约 3.5GB）
            - 模型会缓存在 ~/.cache/huggingface/ 目录
            - 使用双重检查锁定模式，支持并发调用
        """
        if cls._model is not None:
            logger.warning(f"Model already initialized: {cls._model_type}-{cls._model_size} on {cls._device}")
            return

        # 自动检测设备
        if device is None:
            device = cls.get_available_device()

        # 选择最优数据类型
        dtype = cls.get_optimal_dtype(device)

        # 构建模型名称（HuggingFace 格式）
        model_name = f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"

        logger.info(f"Loading Qwen3-TTS model: {model_name}")
        logger.info(f"Device: {device}, dtype: {dtype}")

        async with cls._lock:
            # 双重检查，防止并发初始化
            if cls._model is not None:
                return

            try:
                # 动态导入（避免启动时就依赖 qwen-tts）
                from qwen_tts import Qwen3TTSModel

                # 加载模型
                cls._model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map=device,
                    dtype=dtype,
                    # attn_implementation="flash_attention_2",  # 如果安装了 FlashAttention 2，启用它以减少内存占用
                )
                cls._model_type = model_type
                cls._model_size = model_size
                cls._device = device

                logger.info("✓ Qwen3-TTS model loaded successfully")
                logger.info(f"  Model: {model_name}")
                logger.info(f"  Device: {device}")
                logger.info(f"  Memory: {cls._get_memory_info()}")

            except ImportError as e:
                logger.error(f"Failed to import qwen_tts: {e}")
                logger.error("Please install: pip install qwen-tts")
                raise RuntimeError("qwen-tts package not installed. Run: pip install qwen-tts")
            except Exception as e:
                logger.error(f"Failed to load Qwen3-TTS model: {e}", exc_info=True)
                raise

    @classmethod
    def _get_memory_info(cls) -> str:
        """获取内存信息"""
        try:
            if cls._device == "mps":
                # MPS 不直接提供显存信息
                import psutil
                mem = psutil.virtual_memory()
                return f"System RAM: {mem.total / (1024**3):.1f}GB (Available: {mem.available / (1024**3):.1f}GB)"
            elif cls._device.startswith("cuda"):
                allocated = torch.cuda.memory_allocated(cls._device) / (1024**3)
                reserved = torch.cuda.memory_reserved(cls._device) / (1024**3)
                total = torch.cuda.get_device_properties(cls._device).total_memory / (1024**3)
                return f"VRAM: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.1f}GB total"
            else:
                import psutil
                mem = psutil.virtual_memory()
                return f"RAM: {mem.total / (1024**3):.1f}GB (Available: {mem.available / (1024**3):.1f}GB)"
        except Exception:
            return "Unknown"

    @classmethod
    async def get_model(cls) -> "Qwen3TTSModel":
        """获取已加载的模型"""
        if cls._model is None:
            raise RuntimeError(
                "Qwen3-TTS model not initialized. "
                "Call initialize() first or check if QWEN_ENABLE=true in config."
            )
        return cls._model

    @classmethod
    async def generate_custom_voice(
        cls,
        text: str,
        speaker: str = "Vivian",
        language: Language = "Chinese",
        instruct: Optional[str] = None,
        max_new_tokens: int = 2048,
    ) -> tuple[bytes, int]:
        """
        使用 CustomVoice 模式生成语音

        Args:
            text: 要转换的文本
            speaker: 说话人名称（从 CUSTOM_VOICE_SPEAKERS 中选择）
            language: 语言
            instruct: 可选的风格指令
            max_new_tokens: 最大生成 token 数

        Returns:
            (audio_bytes, sample_rate): WAV 格式的音频字节数据和采样率
        """
        model = await cls.get_model()

        if cls._model_type != "CustomVoice":
            raise RuntimeError(
                f"Current model is {cls._model_type}, please initialize with model_type='CustomVoice'"
            )

        if speaker not in CUSTOM_VOICE_SPEAKERS:
            raise ValueError(
                f"Invalid speaker: {speaker}. "
                f"Available speakers: {', '.join(CUSTOM_VOICE_SPEAKERS)}"
            )

        try:
            import soundfile as sf
            import numpy as np

            logger.info(f"Generating speech: speaker={speaker}, lang={language}, text_len={len(text)}")

            # 生成语音
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                max_new_tokens=max_new_tokens,
            )

            # 将 numpy 数组转换为 WAV 格式的字节流
            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sr, format='WAV')
            audio_bytes = buffer.getvalue()
            buffer.close()

            logger.info(f"✓ Generated {len(audio_bytes)/1024:.1f}KB audio at {sr}Hz")
            return audio_bytes, sr

        except Exception as e:
            logger.error(f"Error in generate_custom_voice: {e}", exc_info=True)
            raise

    @classmethod
    async def generate_voice_design(
        cls,
        text: str,
        instruct: str,
        language: Language = "Chinese",
        max_new_tokens: int = 2048,
    ) -> tuple[bytes, int]:
        """
        使用 VoiceDesign 模式生成语音（自然语言描述声音）

        Args:
            text: 要转换的文本
            instruct: 声音描述（如："用温柔的声音说"）
            language: 语言
            max_new_tokens: 最大生成 token 数

        Returns:
            (audio_bytes, sample_rate): WAV 格式的音频字节数据和采样率
        """
        model = await cls.get_model()

        if cls._model_type != "VoiceDesign":
            raise RuntimeError(
                f"Current model is {cls._model_type}, please initialize with model_type='VoiceDesign'"
            )

        try:
            import soundfile as sf

            logger.info(f"Generating voice design: lang={language}, text_len={len(text)}")

            wavs, sr = model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
                max_new_tokens=max_new_tokens,
            )

            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sr, format='WAV')
            audio_bytes = buffer.getvalue()
            buffer.close()

            logger.info(f"✓ Generated {len(audio_bytes)/1024:.1f}KB audio at {sr}Hz")
            return audio_bytes, sr

        except Exception as e:
            logger.error(f"Error in generate_voice_design: {e}", exc_info=True)
            raise

    @classmethod
    async def generate_voice_clone(
        cls,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: Language = "Chinese",
        x_vector_only: bool = False,
        max_new_tokens: int = 2048,
    ) -> tuple[bytes, int]:
        """
        使用 Base 模型进行声音克隆

        Args:
            text: 要转换的文本
            ref_audio_path: 参考音频文件路径
            ref_text: 参考音频的文字内容
            language: 语言
            x_vector_only: 是否仅使用 x-vector（不需要 ref_text，但质量会降低）
            max_new_tokens: 最大生成 token 数

        Returns:
            (audio_bytes, sample_rate): WAV 格式的音频字节数据和采样率
        """
        model = await cls.get_model()

        if cls._model_type != "Base":
            raise RuntimeError(
                f"Current model is {cls._model_type}, please initialize with model_type='Base'"
            )

        try:
            import soundfile as sf

            logger.info(f"Generating voice clone: lang={language}, x_vector_only={x_vector_only}")

            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_path,
                ref_text=ref_text if not x_vector_only else None,
                x_vector_only_mode=x_vector_only,
                max_new_tokens=max_new_tokens,
            )

            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sr, format='WAV')
            audio_bytes = buffer.getvalue()
            buffer.close()

            logger.info(f"✓ Generated {len(audio_bytes)/1024:.1f}KB audio at {sr}Hz")
            return audio_bytes, sr

        except Exception as e:
            logger.error(f"Error in generate_voice_clone: {e}", exc_info=True)
            raise

    @classmethod
    @lru_cache()
    def get_supported_speakers(cls) -> list[str]:
        """获取支持的说话人列表"""
        return CUSTOM_VOICE_SPEAKERS.copy()

    @classmethod
    @lru_cache()
    def get_supported_languages(cls) -> list[str]:
        """获取支持的语言列表"""
        return ["Auto", "Chinese", "English", "Japanese", "Korean",
                "French", "German", "Spanish", "Portuguese", "Russian"]

    @classmethod
    def get_speaker_description(cls, speaker: str) -> str:
        """获取说话人描述"""
        return SPEAKER_DESCRIPTIONS.get(speaker, "Unknown speaker")

    @classmethod
    def get_model_info(cls) -> dict:
        """获取当前模型信息"""
        return {
            "model_type": cls._model_type,
            "model_size": cls._model_size,
            "device": cls._device,
            "initialized": cls._model is not None,
        }

    @classmethod
    async def health_check(cls) -> dict:
        """健康检查"""
        try:
            model = await cls.get_model()
            return {
                "status": "healthy",
                "model": cls.get_model_info(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": cls.get_model_info(),
            }


@asynccontextmanager
async def get_qwen_model():
    """上下文管理器，用于确保模型在使用前已初始化"""
    try:
        yield await QwenTTSEngine.get_model()
    except RuntimeError as e:
        logger.error(f"Qwen model not initialized: {e}")
        raise
