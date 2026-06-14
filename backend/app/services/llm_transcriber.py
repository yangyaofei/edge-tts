from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """你是一个专业的中文播报稿编辑。将输入文本转化为适合语音合成的口语化播报稿。

规则：
1. 去掉所有 Markdown 格式符号（#、*、>、|、- 等），保留语义内容
2. 数字转中文口语（996→九百九十六，15:30→三点三十）
3. 英文缩写按语境处理：WWDC→W W D C，API→接口 或 A P I
4. 文件名/路径展开（config.json→config 点 J S O N）
5. 多音字用拼音标注正确读音（银行→银háng，行走→xíng走，重新→chóng新）
6. 表格/列表转为连贯口语，加自然衔接词
7. 保留段落结构（用空行分段），不要合并成一大段
8. 不要加"大家好"之类的客套话，直接转写内容
9. 不要加解释或注释，只输出转写后的文本

多音字标注格式：在需要纠正读音的字后面用括号标注拼音，例如：银行(háng)
如果该字在上下文中读音明确（模型能正确判断），就不需要标注。"""


class LLMTranscriber:
    """LLM 转写器：将原始 Markdown 转为口语化播报稿。

    使用 OpenAI-compatible API（支持 OpenAI / DeepSeek / 本地模型）。
    这是 pipeline 的可选前置步骤，独立于 TextPreprocessor（正则清理）。
    """

    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        prompt: str = DEFAULT_PROMPT,
        timeout: float = 60.0,
        max_tokens: int = 8192,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.prompt = prompt
        self.timeout = timeout
        self.max_tokens = max_tokens

    def is_configured(self) -> bool:
        """是否已配置可用。"""
        return bool(self.api_url and self.api_key)

    async def transcribe(self, text: str) -> str:
        """调用 LLM 将文本转为口语化播报稿。

        Args:
            text: 原始 Markdown 文本

        Returns:
            转写后的口语化文本
        """
        if not self.is_configured():
            logger.warning("LLMTranscriber not configured, returning original text")
            return text

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": text},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.3,
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        logger.info(f"LLM transcribe: {len(text)} chars → model={self.model}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.api_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                result = data["choices"][0]["message"]["content"]
                logger.info(f"LLM transcribe done: {len(result)} chars")
                return result
        except Exception as e:
            logger.error(f"LLM transcribe failed: {e}, returning original text")
            return text
