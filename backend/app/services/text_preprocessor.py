from __future__ import annotations

import re


class TextPreprocessor:
    """通用文本预处理：Markdown → 口语化文本。所有 TTS engine 共享。

    可独立开关每一步，正交组合。
    """

    def process(self, text: str) -> str:
        if not text:
            return ""
        text = self.strip_markdown(text)
        text = self.normalize_symbols(text)
        text = self.normalize_numbers(text)
        text = self.normalize_english(text)
        text = self.cleanup_whitespace(text)
        return text.strip()

    def strip_markdown(self, text: str) -> str:
        """去掉 Markdown 格式符号，保留语义文本。"""
        # 代码块 ```...``` → 移除（代码不适合朗读）
        text = re.sub(r"```[\s\S]*?```", "", text)

        # 行内代码 `code` → code
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # 图片 ![alt](url) → alt
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

        # 链接 [text](url) → text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # 标题标记 # ## ### → 移除
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # 引用 > → 移除
        text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)

        # 粗体/斜体 **text** *text* __text__ _text_ → text
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
        text = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"\1", text)

        # 删除线 ~~text~~ → text
        text = re.sub(r"~~([^~]+)~~", r"\1", text)

        # 水平分割线 --- *** ___ → 移除
        text = re.sub(r"^[\-\*_]{3,}\s*$", "", text, flags=re.MULTILINE)

        # 无序列表标记 - * + → 移除
        text = re.sub(r"^\s*[\-\*\+]\s+", "", text, flags=re.MULTILINE)

        # 有序列表标记 1. 2. → 移除
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

        # 任务列表 [x] [ ] → 移除
        text = re.sub(r"^\s*\[[ xX]\]\s*", "", text, flags=re.MULTILINE)

        # 表格：移除分隔行 |---|---|，保留内容
        text = re.sub(r"^\|?[\s\-:|]+\|?\s*$", "", text, flags=re.MULTILINE)
        # 表格单元格分隔 | → 逗号
        text = re.sub(r"\s*\|\s*", "，", text)

        # HTML 标签 → 移除
        text = re.sub(r"<[^>]+>", "", text)

        # HTML 实体
        entities = {
            "&amp;": "&", "&lt;": "<", "&gt;": ">",
            "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)

        return text

    def normalize_symbols(self, text: str) -> str:
        """特殊符号转口语表达。"""
        symbol_map = {
            "→": "到",
            "←": "从",
            "↑": "上升",
            "↓": "下降",
            "©": "版权",
            "®": "注册商标",
            "™": "商标",
            "@": "艾特",
            "&": "和",
            "~": "约",
            "≈": "约等于",
            "≠": "不等于",
            "≤": "小于等于",
            "≥": "大于等于",
            "∞": "无穷",
            "°": "度",
            "×": "乘以",
            "÷": "除以",
            "±": "正负",
            "√": "根号",
            "¶": "段落",
            "•": "，",
            "…": "。",
        }
        for symbol, word in symbol_map.items():
            text = text.replace(symbol, word)

        # 箭头组合
        text = text.replace("=>", "推导出")
        text = text.replace("->", "到")

        return text

    def normalize_numbers(self, text: str) -> str:
        """数字/时间/日期转口语表达。"""
        # 时间 HH:MM:SS → X点X分X秒
        text = re.sub(
            r"(\d{1,2}):(\d{2}):(\d{2})",
            lambda m: f"{int(m.group(1))}点{int(m.group(2))}分{int(m.group(3))}秒",
            text,
        )

        # 时间 HH:MM → X点X分
        text = re.sub(
            r"(\d{1,2}):(\d{2})",
            lambda m: f"{int(m.group(1))}点{int(m.group(2))}分",
            text,
        )

        # 日期 YYYY-MM-DD → X年X月X日
        text = re.sub(
            r"(\d{4})-(\d{1,2})-(\d{1,2})",
            lambda m: f"{int(m.group(1))}年{int(m.group(2))}月{int(m.group(3))}日",
            text,
        )

        # 版本号 v1.2.3 → V 一点二点三（保持不拆，交给 engine）
        # 百分比 50% → 五十百分点（交给 engine 或 LLM 处理更自然）
        # 其他数字保持原样，由 TTS engine 处理

        return text

    def normalize_english(self, text: str) -> str:
        """英文缩写/文件扩展名展开为字母拼写。"""
        # 文件扩展名 .md .json .py .txt → 点 M D / 点 J S O N
        ext_map = {
            ".md": " 点 M D ",
            ".MD": " 点 M D ",
            ".json": " 点 J S O N ",
            ".JSON": " 点 J S O N ",
            ".js": " 点 J S ",
            ".JS": " 点 J S ",
            ".py": " 点 P Y ",
            ".PY": " 点 P Y ",
            ".txt": " 点 T X T ",
            ".TXT": " 点 T X T ",
            ".ts": " 点 T S ",
            ".TS": " 点 T S ",
            ".csv": " 点 C S V ",
            ".xml": " 点 X M L ",
            ".html": " 点 H T M L ",
            ".yaml": " 点 Y A M L ",
            ".yml": " 点 Y M L ",
            ".sh": " 点 S H ",
            ".sql": " 点 S Q L ",
            ".wav": " 点 W A V ",
            ".mp3": " 点 M P 三 ",
        }
        for ext, replacement in ext_map.items():
            text = text.replace(ext, replacement)

        return text

    def cleanup_whitespace(self, text: str) -> str:
        """清理多余空白。"""
        # 多个空行压缩为一个
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 行首行尾空格
        text = re.sub(r"^[ \t]+", "", text, flags=re.MULTILINE)
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        # 多个空格压缩
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text
