"""
文本分块服务模块

本模块提供文本分块功能，将长文本分割成适合 TTS 处理的短片段。

用途：
- 长文本 TTS 处理（避免超时）
- 文本分段朗读（提高自然度）
- 音频文件管理（按段落生成）

支持的策略：
- paragraph: 按段落分割（基于换行符）
- sentence: 按句子分割（基于句号等，待实现）
- punctuation: 按标点符号分割（待实现）
"""

from typing import List


class TextChunker:
    """
    文本分块器类

    提供静态方法用于将文本按不同策略分割成多个块。
    """

    @staticmethod
    def chunk_text(text: str, strategy: str = "paragraph") -> List[str]:
        """
        将文本分割成多个块

        Args:
            text: 要分块的文本
            strategy: 分块策略
                - "paragraph": 按段落分割（推荐，已实现）
                - "sentence": 按句子分割（未实现）
                - "punctuation": 按标点符号分割（未实现）

        Returns:
            List[str]: 文本块列表

        Example:
            >>> text = "第一段。\\n\\n第二段。\\n\\n第三段。"
            >>> chunks = TextChunker.chunk_text(text, "paragraph")
            >>> print(chunks)
            ['第一段。', '第二段。', '第三段。']

        Note:
            - 空文本返回空列表
            - 段落策略：按换行符（\\n）分割
            - 自动过滤空白段落
            - 每个段落会去除首尾空白字符

        Todo:
            - 实现 sentence 策略（按句号、问号、感叹号分割）
            - 实现 punctuation 策略（按逗号、分号等分割）
            - 支持自定义分隔符
        """
        # 空文本处理
        if not text:
            return []

        # 去除首尾空白
        text = text.strip()

        # 段落分割策略（已实现）
        if strategy == "paragraph":
            # 按换行符分割
            # - 先规范化换行符（\\r\\n -> \\n）
            # - 按换行符分割
            # - 过滤空白段落
            # - 去除每个段落的首尾空白
            normalized_text = text.replace("\r\n", "\n")
            paragraphs = [p.strip() for p in normalized_text.split("\n") if p.strip()]
            return paragraphs

        # 其他策略（未实现，返回原文本）
        # Todo: 'sentence' or 'punctuation' strategy
        return [text]
