from __future__ import annotations

import re


def split_sentences(text: str, min_len: int = 4) -> list[str]:
    """按标点/换行分句, **完整保留所有字符**, 拼接结果恒等于输入文本。

    目的 (purpose): 给 /segment 接口提供确定性的句子边界, 让前端能做
    original↔tts_text 逐句对齐和 DOM 高亮 (original 拼起来 == 原文)。
    做法 (how): lookbehind 切分 (不消费分隔符), 短句向下合并到 min_len。
    与 TextChunker 的区别: TextChunker 的 sentence 策略会吃掉分隔符
    (拼接丢失原文), 这里保留。

    不对段做 strip —— 否则换行/空白会丢失, 破坏 "拼接 == 原文" 的不变量。
    空白处理交给调用方 (前端 DOM / TTS 引擎各自 normalize)。
    """
    if not text:
        return []
    if not text.strip():
        return []
    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    out: list[str] = []
    buf = ""
    for p in parts:
        if p == "":
            continue
        buf = p if not buf else buf + p
        if len(buf) >= min_len:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out if out else [text]


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
