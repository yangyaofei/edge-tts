"""
Edge TTS 引擎模块

本模块封装了 Microsoft Edge TTS (Text-to-Speech) 的功能。
Edge TTS 是 Microsoft 提供的免费在线语音合成服务。

特点：
- 支持 100+ 种语音和方言
- 返回 MP3 格式音频流
- 支持语速和音调调整
- 无需 API 密钥
"""

import edge_tts
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


class EdgeTTSEngine:
    """
    Edge TTS 引擎类

    提供 Edge TTS 的静态方法接口，包括：
    - 获取可用语音列表
    - 生成流式音频
    """

    @staticmethod
    async def get_voices() -> list:
        """
        获取所有可用的 Edge TTS 语音列表

        Returns:
            list: 语音信息列表，每个语音包含：
                - Name: 语音 ID（如 zh-CN-XiaoxiaoNeural）
                - FriendlyName: 显示名称
                - Gender: 性别
                - Locale: 地区/语言

        Example:
            >>> voices = await EdgeTTSEngine.get_voices()
            >>> for voice in voices:
            ...     print(f"{voice['Name']}: {voice['FriendlyName']}")
        """
        voices = await edge_tts.list_voices()
        return voices

    @staticmethod
    async def generate_stream(
        text: str,
        voice: str,
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ) -> AsyncGenerator[bytes, None]:
        """
        生成音频流（MP3 格式）

        Args:
            text: 要转换的文本
            voice: 语音 ID（如 zh-CN-XiaoxiaoNeural）
            rate: 语速调整
                - "+0%": 正常速度
                - "-10%": 慢 10%
                - "+20%": 快 20%
            pitch: 音调调整
                - "+0Hz": 正常音调
                - "-10Hz": 低 10Hz
                - "+10Hz": 高 10Hz

        Yields:
            bytes: MP3 格式的音频数据块

        Example:
            >>> async for chunk in EdgeTTSEngine.generate_stream("你好", "zh-CN-XiaoxiaoNeural"):
            ...     # 处理音频块
            ...     pass

        Note:
            这是一个异步生成器，产生的音频块可以直接用于 HTTP 流式响应。
        """
        logger.debug(f"Starting generation for text: {text[:20]}...")
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)

        count = 0
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    count += 1
                    # logger.debug(f"Yielding audio chunk {count}, size={len(chunk['data'])}")
                    yield chunk["data"]
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            raise e

        logger.debug(f"Generation finished. Total audio chunks: {count}")
