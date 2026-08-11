from __future__ import annotations

import logging

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.core.config import settings

logger = logging.getLogger(__name__)


# 融合 Qwen3-TTS 前处理调研报告规则 + 一般人交流习惯的归一化 prompt。
# 注意: deepseek-v4-flash 是 thinking 模型, 不支持 tool_choice (pydantic-ai 结构化 tool calling),
# 故用 output_type=str 纯文本输出, 靠 prompt 强约束返回格式。
SYSTEM_PROMPT = """你是中文语音合成(TTS)文本归一化专家。把输入的单个句子转成适合朗读的中文文本。

## 核心原则
符合一般人的口语交流习惯,让 TTS 听起来自然、清晰、不会读错。保持原意不变,只调整朗读形式。

## 1. 数字转中文读法
- 年份(数字紧跟"年"): 逐字读。2026年 → 二零二六年;1992年 → 一九九二年
- 数量/基数: 按千百十读法。126 → 一百二十六;2026个 → 两千零二十六个;0.53 → 零点五三
- 数字"2"跟量词: 读"两"。2个 → 两个;2台 → 两台;2人 → 两人
- 小数: 3.14 → 三点一四;0.5 → 零点五
- 分数/百分: 1/5 → 五分之一;6.3% → 百分之六点三;100% → 百分之百
- 序号(如"第3"): 第三
- 电话/长ID(5位以上连续纯数字,非数量非年份): 空格逐字读,其中数字 1 读"幺"。13800000000 → 一 三 八 零 零 零 零 零 零 零 零

## 2. 时间与日期
- 冒号时间: 14:30 → 十四点三十分;8:00 → 八点;09:05 → 九点零五分
- 日期: 2024-03-15 → 二零二四年三月十五日

## 3. 英文缩写与单位(关键:字母间必须加空格,否则 TTS 会连读成错误单词)
- 缩写字母逐字大写,用空格分隔: CPU → C P U;GPU → G P U;URL → U R L;API → A P I
- 单位:字母部分逐字 + 中文量词。GB → G B;MB → M B;GHz → G 赫兹;MHz → M 赫兹;Mbps → M b p s;ms → 毫秒
- 纯英文单词(非缩写)保持原样,不要拆字母: Python、Java、Docker 保持不变

## 4. 多音字(必须做,在字后用括号标注带声调的拼音)
- 银行 → 银行(háng);行程 → 行(xíng)程;重新 → 重新(chóng);重量 → 重量(zhòng)
- 长度 → 长(cháng)度;成长 → 成长(zhǎng);重要 → 重(zhòng)要
- 声调符号: ā á ǎ à | ē é ě è | ī í ǐ ì | ō ó ǒ ò | ū ú ǔ ù | ǖ ǘ ǚ ǜ
- 只标注容易读错的多音字,普通字不标

## 5. 符号与标记清洗
- markdown 标记全部去除: **加粗** → 加粗; `代码` → 代码; # 标题 → 标题; > 引用 → 引用; - 列表 → 列表
- 破折号(——)替换为逗号
- 省略号(……或…)保留,作为语音停顿
- 多余空白压缩为单个空格

## 6. 断句与节奏(重要,否则 TTS 连读听不懂)
- 词的边界必须清晰,特别是英文缩写字母之间必须加空格(CPU 不加空格会被 TTS 当成一个单词连读,听不懂)
- 保留所有句号、逗号、问号、感叹号、分号、冒号作为停顿
- 过长的无标点分句(超过约 20 个字),在语义合适处插入逗号便于换气
- 输出仍是单个句子结构,只是内部节奏更合理

## 输出格式(严格遵守)
只输出归一化后的朗读文本本身,一个句子。不要加任何解释、注释、前后缀、引号、JSON、markdown 代码块。直接输出文本内容。"""


_agent: Agent | None = None
_disabled: bool = False


def get_normalize_agent() -> Agent | None:
    """单例 Agent。未配置 API 时返回 None。"""
    global _agent, _disabled
    if _agent is not None:
        return _agent
    if _disabled:
        return None
    api_url = settings.TTS_LLM_TRANSCRIBE_API_URL
    api_key = settings.TTS_LLM_TRANSCRIBE_API_KEY
    model_name = settings.TTS_LLM_TRANSCRIBE_MODEL
    if not (api_url and api_key and model_name):
        _disabled = True
        logger.warning("normalize agent disabled: TTS_LLM_TRANSCRIBE_* not configured")
        return None
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(api_key=api_key, base_url=api_url),
    )
    _agent = Agent(model, output_type=str, system_prompt=SYSTEM_PROMPT)
    return _agent


async def normalize_sentence(text: str) -> str:
    """单句归一化。返回适合 TTS 的文本。失败时抛异常(由调用方决定降级策略)。"""
    agent = get_normalize_agent()
    if agent is None:
        return text
    result = await agent.run(text)
    tts_text = result.output.strip() if isinstance(result.output, str) else str(result.output).strip()
    return tts_text or text
