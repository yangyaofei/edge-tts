from __future__ import annotations

import re


class TextChunker:
    """通用文本分段器。支持段落、句子、固定字符数三种策略。"""

    SENTENCE_ENDINGS = re.compile(r"[。！？!？\.\n]+")

    @staticmethod
    def chunk_text(
        text: str,
        strategy: str = "paragraph",
        max_chars: int = 500,
    ) -> list[str]:
        """将文本切分为 chunks。

        Args:
            text: 输入文本
            strategy: 切分策略
                - paragraph: 按空行分段，超长段落按句子再切
                - sentence: 按句号/问号/感叹号切
                - fixed: 固定字符数
            max_chars: 单段最大字符数，超长则再切

        Returns:
            chunk 文本列表
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        if strategy == "paragraph":
            return TextChunker._chunk_by_paragraph(text, max_chars)
        elif strategy == "sentence":
            return TextChunker._chunk_by_sentence(text, max_chars)
        elif strategy == "fixed":
            return TextChunker._chunk_fixed(text, max_chars)
        else:
            return [text]

    @staticmethod
    def _chunk_by_paragraph(text: str, max_chars: int) -> list[str]:
        """按空行分段，超长段落按句子再切。"""
        normalized = text.replace("\r\n", "\n")
        raw_paragraphs = re.split(r"\n\s*\n", normalized)
        paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

        result: list[str] = []
        for para in paragraphs:
            if len(para) <= max_chars:
                result.append(para)
            else:
                result.extend(TextChunker._chunk_by_sentence(para, max_chars))

        return result

    @staticmethod
    def _chunk_by_sentence(text: str, max_chars: int) -> list[str]:
        """按句子切分，短句合并到 max_chars 以内。"""
        sentences = TextChunker.SENTENCE_ENDINGS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        result: list[str] = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_chars:
                current = (current + "，" + sent) if current else sent
            else:
                if current:
                    result.append(current)
                if len(sent) <= max_chars:
                    current = sent
                else:
                    result.extend(TextChunker._chunk_fixed(sent, max_chars))
                    current = ""

        if current:
            result.append(current)

        return result

    @staticmethod
    def _chunk_fixed(text: str, max_chars: int) -> list[str]:
        """固定字符数切分。"""
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
