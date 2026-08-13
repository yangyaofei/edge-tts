from __future__ import annotations

import re

from markdown_it import MarkdownIt


def split_markdown(text: str, max_len: int = 500, min_len: int = 30) -> list[str]:
    """用 markdown AST 按块级元素分割, 保留原始 markdown 源码。

    目的: 确定性分割 (按 markdown 语法树), 不做格式修正。
    每个块级元素 (标题/段落/列表项/引用/表格/代码块) 成一个段落。
    markdown 标记保留原样, 交给 normalize 阶段的 LLM 处理。
    超长段落按句号二次分割。
    标题段合并到后续第一个非标题段 (避免标题单独播放不流畅)。
    非标题段落保持独立, 不互相合并。
    """
    if not text or not text.strip():
        return []

    # 预处理: 修复 "#\n标题文本" → "# 标题文本"
    # 上游生成的文章把 # 和标题文本分在两行, markdown_it 会把 # 解析成空标题。
    text = re.sub(r'^(#{1,6})\s*\n([^\n#]+)$', r'\1 \2', text, flags=re.MULTILINE)

    md = MarkdownIt("commonmark")
    tokens = md.parse(text)
    lines = text.split("\n")

    segments: list[str] = []
    headings: list[bool] = []
    depth = 0

    for token in tokens:
        pre_depth = depth
        depth += token.nesting  # +1 open, -1 close, 0 self-closing

        # 跳过水平线
        if token.type == "hr":
            continue

        # 顶层自闭合块 (代码块)
        if token.type in ("fence", "code_block") and pre_depth == 0:
            content = token.content.strip()
            if content:
                segments.append(content)
                headings.append(False)
            continue

        # 顶层块级元素: 标题/段落/引用/表格 -- 用 map 取源码
        if (
            token.nesting == 1
            and pre_depth == 0
            and token.type
            in ("heading_open", "paragraph_open", "blockquote_open", "table_open")
        ):
            raw = _source_lines(lines, token.map)
            if raw:
                segments.append(raw)
                headings.append(token.type == "heading_open")
            continue

        # 列表项 (depth 1, 在 list 内部) -- 每项独立
        if token.nesting == 1 and pre_depth == 1 and token.type == "list_item_open":
            raw = _source_lines(lines, token.map)
            if raw:
                segments.append(raw)
                headings.append(False)
            continue

    # 拆超长段落 (标题不拆), 再合并标题到后续段
    split: list[str] = []
    split_heads: list[bool] = []
    for seg, is_head in zip(segments, headings):
        if len(seg) > max_len and not is_head:
            parts = _split_long(seg, max_len)
            split.extend(parts)
            split_heads.extend([False] * len(parts))
        else:
            split.append(seg)
            split_heads.append(is_head)
    return _merge_headings(split, split_heads)


def _merge_headings(segments: list[str], headings: list[bool]) -> list[str]:
    """标题段合并到后续第一个非标题段。

    段落级粒度: 非标题段落保持独立, 不互相合并。
    连续标题累积, 遇到段落时一并合并。末尾标题独立。
    """
    merged: list[str] = []
    pending: list[str] = []
    for seg, is_head in zip(segments, headings):
        if is_head:
            pending.append(seg)
        else:
            if pending:
                merged.append("\n".join(pending + [seg]))
                pending = []
            else:
                merged.append(seg)
    if pending:
        merged.extend(pending)
    return merged


def _source_lines(lines: list[str], mapping: list[int] | None) -> str:
    """从 token.map [start, end) 提取原始源码行。"""
    if not mapping:
        return ""
    start, end = mapping
    return "\n".join(lines[start:end]).strip()


def _split_long(text: str, max_len: int) -> list[str]:
    """超长段落按强句末标点(句号/感叹/问号)分割, 短句合并到 max_len 以内。
    不在分号、换行处切割——它们是句内停顿, 切了会产生碎片。"""
    parts = re.split(r"(?<=[。！？!?])", text)
    result: list[str] = []
    current = ""
    for p in parts:
        if not p:
            continue
        if len(current) + len(p) > max_len and current:
            result.append(current.strip())
            current = p
        else:
            current += p
    if current.strip():
        result.append(current.strip())
    return result


def split_sentences(text: str, min_len: int = 4) -> list[str]:
    """按标点/换行分句 (旧接口, 保留兼容)。"""
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
