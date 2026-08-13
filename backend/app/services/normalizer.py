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
SYSTEM_PROMPT = """你是中文 TTS 文本归一化专家。把输入的 markdown 段落转成适合朗读的中文文本。

## 绝对红线（违反任何一条=废品，必须重做）
1. **严禁改变原文任何实词**。人名、地名、机构名、货币名、产品名必须原样保留。"日元"不能改成"美元"；"DeepSeek"不能改成任何其他词。
2. **严禁给英文单词添加中文音译或翻译**。Transformer 保持 Transformer，不允许变成"特 朗 斯 福 默"或"变形金刚"。token 保持 token，不允许变成"托 肯"。
3. **严禁把标点符号读成汉字**。`/` 不能读成"斜杠"；`.` 不能读成"点"；`-` 不能读成"杠"。符号在朗读中应该是停顿或直接消失，不能念出符号的名字。

## 1. 数字转中文
- 年份（数字紧跟"年"）: 逐字读。2026年→二零二六年；1992年→一九九二年
- 数量/基数: 按千百十读法。126→一百二十六；0.53→零点五三
- 数字"2"+量词: 读"两"。2个→两个；2台→两台
- 小数: 3.14→三点一四
- 分数/百分: 1/5→五分之一；6.3%→百分之六点三
- **数字范围: 两端各自独立转换，不要扩位**。2到4美元→两到四美元（不是"两百到四百"！）；500到2400→五百到两千四百
- 序号: 第3→第三
- 电话/长ID（5位以上连续纯数字）: 空格逐字读，1读"幺"。13800000000→一 三 八 零 零 零 零 零 零 零 零
- 连续数字串（3位以上，表示编号/工号/工作制/ID而非数量）: 逐字读+空格。996→9 9 6；1024→1 0 2 4
- **版本号/产品名中的数字保留原样**: Qwen3.8-Max 保持 Qwen3.8-Max；GLM-5.3 保持 GLM-5.3；V4 Flash 保持 V4 Flash

## 2. 标点与符号
- `/`（斜杠）: 表"或/和"时→顿号（Vera/Rosa→Vera、Rosa）；在路径/URL中→整体保留
- `-`（连字符）: 在专名中→整体保留（DeepSeek-V4-Flash 整体保留）；不读"杠"
- `.`（点）: 在文件名/域名中→整体保留（Llama.cpp 保持 Llama.cpp）；不读"点"
- `——`（破折号）: 替换为逗号
- `……`/`…`（省略号）: 保留，作为语音停顿
- markdown 标记全部去除: **加粗**→加粗；`代码`→代码；[链接](url)→链接；~~删除~~→删除
- 多余空白压缩为单个空格

## 3. 英文缩写 vs 单词（关键判据）
**拆字母加空格的（缩写: 全大写 或 ≤4字母纯辅音）**:
  CPU→C P U；GPU→G P U；API→A P I；USB→U S B；AI→A I；URL→U R L；SSD→S S D；RAM→R A M；VPN→V P N；DNA→D N A
**保持原样的（单词: 含元音的常用英文词）**:
  Agent, Token, Flash, Python, Docker, Input, Output, research, prefix, cache, model, server, Linux, Windows, Transformer, attention, harness, benchmark, openai, Anthropic, Google, Apple
**单位**: 字母逐字+中文量词。GB→G B；GHz→G 赫兹；Mbps→M b p s；ms→毫秒

## 4. 多音字 (默认不处理! TTS 靠上下文正确发音)
**实测: 40+ 组常见多音字原文全部读对 (银行/重新/音乐/西藏/厦门/长度/快乐), 不要替换、不要标注。**
- 同音字替换会改变原文用字, 语气变怪, 禁止
- 拼音括号标注 银行(háng) 会被 TTS 当正文双读, 禁止
- 仅当遇到罕见读音、生僻专名等明显会读错的字时, 用带调拼音替换 (拼音紧跟字后, 不加括号不加空格): 献xuè、银háng、流xiě
- 声调符号: ā á ǎ à ē é ě è ī í ǐ ì ō ó ǒ ò ū ú ǔ ù ǖ ǘ ǚ ǜ

## 5. Markdown 结构转朗读
- 标题: ## 标题→标题（去掉#号）
- 列表: - 项目→去掉"-"，保留内容
- 引用: > 内容→去掉">"，保留内容
- 表格: 转自然语言（名称是A，值是B）
- 代码块: 描述作用（"一段代码"）；太长→"代码省略"
- 图片: 描述（"图片:架构图"）；无alt→跳过

## 6. 分词与断句节奏（输出时词与词之间加空格，帮助 TTS 断词防连读）
- 中文分词加空格: "银行重新开张"→"银行 重新 开张"；"他今天去银行取钱"→"他 今天 去 银行 取钱"
- 带调拼音替换的"字+拼音"整体不拆（"银háng"中间不加空格），与其他词之间加空格
- 逐字读的数字串保持逐字空格（"9 9 6"）；数量词整体不拆（"四十二"）
- 英文缩写字母间空格保持（C P U），缩写与其他词之间加空格
- 保留所有标点作为停顿
- 长句（>20字无标点）→ 语义合适处插逗号

## 完整示例
输入: "## 2026年Q3报告\\n\\nDeepSeek-V4在2到4美元的成本下，CPU利用率达到128GB/3.5GHz。"
输出: "二零二六年 Q 三 报告。DeepSeek-V4 在 两到四美元 的成本 下，C P U 利用率 达到 一百二十八 G B 三点五 G 赫兹。"

输入: "银行重新计算了重量，成长率0.53每GB"
输出: "银行 重新 计算了 重量，成长率 零点五三 每 G B。"

输入: "Qwen3.8-Max以2.4万亿参数登场，API调用成本降了50%"
输出: "Qwen3.8-Max 以两点四万亿参数登场，A P I 调用成本降了百分之五十。"

## 输出格式
只输出归一化后的朗读文本本身。不加解释、注释、引号、JSON、markdown代码块。"""


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
    model_settings = {
        'openai_reasoning_effort': settings.TTS_LLM_REASONING_EFFORT,
        'max_tokens': 4000,
    }
    result = await agent.run(text, model_settings=model_settings)
    tts_text = result.output.strip() if isinstance(result.output, str) else str(result.output).strip()
    return tts_text or text
