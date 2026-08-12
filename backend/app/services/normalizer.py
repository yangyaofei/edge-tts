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

## 4. 多音字 (同音字替换 — TTS 端到端模型不理解拼音标注, 必须替换为单音字)
**原理: Qwen3-TTS 是端到端模型, 没有独立 g2p 层, 不理解 汉字(pinyin) 格式。
所以不能标注拼音, 必须把可能读错的多音字替换为只有一个读音的同音字。**
**输出前自检: 确认没有残留的拼音标注括号。**
替换规则 (根据语境选读音, 替换为单音同音字):
  重(chóng 重复义): 重申→虫申; 重复→虫复; 重新→虫新; 重建→虫建; 重来→虫来
  重(zhòng 沉重义): 重要→仲要; 重量→仲量; 重点→仲点; 严重→严仲
  行(háng 银行义): 银行→银航; 行业→航业; 行内→航内
  行(xíng 行走义): 步行→步形; 行为→形为; 行走→形走; 行动→形动
  长(cháng 长度义): 长度→肠度; 长期→肠期; 长远→肠远
  长(zhǎng 成长义): 成长→成涨; 生长→生涨; 长大→涨大
- 只替换 TTS 容易读错的 (重/行/长/朝/乐/更 等), 常见的不替换 (了/的/得/地)
- 找不到合适单音同音字的, 保持原字不改

## 5. Markdown 结构转朗读
- 标题: ## 标题→标题（去掉#号）
- 列表: - 项目→去掉"-"，保留内容
- 引用: > 内容→去掉">"，保留内容
- 表格: 转自然语言（名称是A，值是B）
- 代码块: 描述作用（"一段代码"）；太长→"代码省略"
- 图片: 描述（"图片:架构图"）；无alt→跳过

## 6. 断句节奏
- 英文缩写字母间必须加空格（CPU→C P U，不加会被TTS连读成单词）
- 保留所有标点作为停顿
- 长句（>20字无标点）→ 语义合适处插逗号

## 完整示例
输入: "## 2026年Q3报告\\n\\nDeepSeek-V4在2到4美元的成本下，CPU利用率达到128GB/3.5GHz。"
输出: "二零二六年 Q 三报告。DeepSeek-V4 在两到四美元的成本下，C P U 利用率达到一百二十八 G B 三点五 G 赫兹。"

输入: "银行重新计算了重量，成长率0.53每GB"
输出: "银行(háng)重新(chóng)计算了重量(zhòng)，成长(zhǎng)率零点五三每 G B。"

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
