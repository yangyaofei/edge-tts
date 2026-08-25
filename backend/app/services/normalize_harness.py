"""TTS 文本归一化 agent harness。

架构 (agent 方式, 非单次调用):
    原句 → 词表命中扫描 → 动态注入 prompt
         → pydantic-ai Agent (每次 run 新建实例, 工具闭包捕获该句 state)
         → 模型自主多轮调用工具:
             submit_edit(find, occurrence, to, rule)  替换编辑
             submit_pause(after, occurrence)          插入停顿
             query_lexicon(word)                      查词表
             check_conservation()                     数字守恒自查
             finish()                                 结束
         → 每个工具调用程序即时校验 (harness):
             find 不存在 / occurrence 越界 / 虚词白名单 / 拼音格式
             → raise ModelRetry 把错误回喂模型自动修正
         → finish 后程序从右到左应用 edits → 数字守恒终检 → tts_text

设计约束:
    - DeepSeek 不支持 tool_choice=required (与 thinking 互斥),
      故不用 output_type=, 而是工具注册 (tool_choice=auto) + finish 工具收尾。
    - 校验失败超过 N 次 ModelRetry 后降级原文, 永不 crash, 永不输出更糟的文本。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.core.config import settings

logger = logging.getLogger(__name__)

LEXICON_PATH = Path(__file__).parent / "tts_lexicon.yaml"

# 虚词/助词黑名单: 这些字多音但 LLM 几乎总是标错 (的→de 灾难), 禁止作为 find 目标
STOPWORD_BLACKLIST = {
    "的", "了", "着", "地", "得", "和", "还", "都", "从", "同", "当",
    "谁", "那", "这", "在", "是", "有", "不", "很", "也", "又", "再",
    "把", "被", "让", "给", "向", "往", "按", "对", "跟", "比",
}

# 拼音内部禁止空格 (gōng yìng 链 是灾难: 双音节拼音被拆开读)
_PINYIN_SPACE_RE = re.compile(
    r"[a-zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]+ [a-zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]+"
)
_DIGIT_RE = re.compile(r"\d+")

SYSTEM_PROMPT = """你是中文语音合成(TTS)文本归一化编辑器。你只提交编辑指令, 不重写全文, 程序负责应用与校验。

## 工作流程
1. 先扫描句子, 找出所有需要处理的内容
2. 用 submit_edit 逐条提交 (find 必须是原文精确子串, 至少 2 个字符)
3. 用 submit_pause 在需要换气停顿的位置插入逗号
4. 不确定的词用 query_lexicon 查询约定读法
5. 全部完成后调用 check_conservation 自查, 再调用 finish 结束

## 转换规则
### 1. 数字转中文读法 (TTS 无法读 ASCII 数字和小数点, 必须全转)
- 年份逐字: 2026年 → 二零二六年
- 数量按位: 126 → 一百二十六; 0.53 → 零点五三
- "2"+量词 → 两: 2个 → 两个
- 小数/版本号必须转: 5.3 → 五点三
- 电话/长ID (5位以上): 空格逐字, 1 读幺: 13800000000 → 幺 三 八 零 零 零 零 零 零 零 零
- 型号里的数字同样转中文: GLM-5.3 → G L M 五点三

### 2. 英文缩写与单位 (先判断读音类型, 再决定拆不拆)
英文词分两类, 处理方式完全不同:
- **缩读词 (acronym, 能当单词拼读)**: 保持原样不拆。YOLO/NASA/Astra/Lyte/rame/.nano
  读作一个词 "you-low"/"na-sa"。判断方法: 元音+辅音交替、人人口语里当词说的。
- **首字母缩写 (initialism, 逐字母拼读)**: 拆开加空格。CPU/GPU/API/URL/LLM/DRAM/ETF
  读作 "C-P-U"/"G-P-U"。判断方法: 辅音连串、口语里逐字母念的。
- 边界情况优先查词表 (query_lexicon), 词表是最高优先级。
- 单位: 字母部分拆开 + 中文量词: GB → G B; GHz → G 赫兹
- 纯英文单词/产品名保持原样: Python, DeepSeek, Claude 不拆
- 型号: 字母部分拆开 + 版本数字转中文: Qwen-3.8 → Q w e n 三点八

### 3. 多音字 (保守!)
- 的/地/得 作结构助词时读轻声 de, TTS 总读错, 统一替换为同音汉字"的" (不加拼音, 无声调拼音会被当字母双读):
  我的书 → 我 的 书; 隐晦地 → 隐晦 的; 跑得快 → 跑 的 快
  即: 助词"地"→"的", 助词"得"→"的", "的"本身保留
  的地得其他用法不替换: 目的dì/的确dí/得手dé
- 上下文能判断的常见多音词: 不处理 (银行/音乐/成长 TTS 自己会读对)
- 只处理歧义高危词, 且必须查询词表 (query_lexicon) 获取标准替换形式
- 其他替换形式: 拼音紧跟被替换的字, 无括号无空格: 重新 → chóng新
- 严禁给 从/同/当/了/着 等虚词标音

### 4. 停顿 (语义单元粒度, 句内也要分块)
- 停顿粒度 = 语义单元 (不只是长句才分): 每个意群之间都要有分界
  例: 苹果 起诉 爆料人 Jon Prosser 的 诉讼 在 取证环节 出现 拖延
  (苹果-起诉-爆料人Jon Prosser的-诉讼-在-取证环节-出现拖延, 每个主谓宾块都分)
- 停顿只放在语义单元边界: 状语之后、主谓之间、并列成分之间
- 英文词/名字 + 中文助词 (里/的/中/上) 结尾的状语, 停顿放助词之后:
  "邀请 Gemini 里 想离开的人" ✓ (Gemini 里 = 状语, 想离开 = 新谓语)
  "邀请 Gemini 里想 离开的人" ✗ (拆散了动宾短语 想离开)
- 动宾短语/谓语内部禁止停顿: 想离开/要做/能吃/会来 不能拆
- 短句内语义块用空格分隔, 意群间用逗号; 长句 (20字以上) 必须有逗号
- 英文缩写与中文之间加空格

### 5. 符号清洗
- markdown 标记去除: **加粗** → 加粗
- 破折号 → 逗号; 省略号保留
- URL/文件路径保留 (模型自己处理)

## 关键约束
- find 必须能在原文找到, 不存在会被拒绝
- 同一片段出现多次用 occurrence 区分 (从 0 开始)
- to 里拼音之间禁止空格 (会被 TTS 拆开读)
- 不要发明原文没有的内容
- 宁可少改, 不可改错"""


@dataclass
class Edit:
    kind: str          # "replace" | "pause"
    find: str
    occurrence: int
    to: str = ""
    rule: str = ""


@dataclass
class HarnessState:
    """单次 run 的可变状态 (每次 normalize_sentence 新建, 并发安全)。"""
    original: str
    edits: list[Edit] = field(default_factory=list)


_lexicon: dict | None = None


def _load_lexicon() -> dict:
    global _lexicon
    if _lexicon is None:
        try:
            _lexicon = yaml.safe_load(LEXICON_PATH.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.warning("lexicon load failed", exc_info=True)
            _lexicon = {}
    return _lexicon


def _matched_entries(text: str) -> list[str]:
    """扫描句子, 返回命中的词表条目 (动态注入 prompt)。"""
    entries = []
    lex = _load_lexicon()
    for category in ("polyphone", "unit", "model"):
        for k, v in (lex.get(category) or {}).items():
            if k and k in text:
                entries.append(f"  {k} → {v}")
    return entries


# ---------- harness 校验 (工具内联) ----------


def _validate_edit(state: HarnessState, find: str, occurrence: int, to: str) -> None:
    """校验单条编辑, 不合格 raise ModelRetry (回喂模型修正)。"""
    n = state.original.count(find)
    if n == 0:
        raise ModelRetry(f"find {find!r} 不在原文中。请从原文复制精确片段。")
    if occurrence < 0 or occurrence >= n:
        raise ModelRetry(f"occurrence={occurrence} 越界 ({find!r} 出现 {n} 次)。")
    if len(find) < 2:
        raise ModelRetry(f"find {find!r} 太短 (<2字符), 无法定位。请包含上下文。")
    if find in STOPWORD_BLACKLIST:
        raise ModelRetry(f"{find!r} 是虚词/常用字, 禁止标注读音。")
    if _PINYIN_SPACE_RE.search(to):
        raise ModelRetry(f"to {to!r} 拼音之间有空格, 会被 TTS 拆开读。拼音必须紧跟汉字。")
    # 符号清洗 (标点→标点/markdown 去除) 合法; 只拦"无中文无字母无标点的纯异常"内容
    if not re.search(r"[\u4e00-\u9fffA-Za-z，。！？；：、,.!?:;\"'“”‘’()\[\]（）]", to):
        raise ModelRetry(f"to {to!r} 无有效文字内容。")


def _apply_edits(original: str, edits: list[Edit]) -> str:
    """应用编辑。replace: 基于原文定位, 从右到左替换, 重叠编辑丢弃后提交的。"""
    # 1. 在原文上定位所有 replace (按提交顺序), 重叠的跳过
    taken: list[tuple[int, int]] = []  # 已占用的 (start, end) 区间
    located: list[tuple[int, int, Edit]] = []
    for e in edits:
        if e.kind != "replace":
            continue
        idx = -1
        for _ in range(e.occurrence + 1):
            idx = original.find(e.find, idx + 1)
            if idx < 0:
                break
        if idx < 0:
            continue
        end = idx + len(e.find)
        if any(s < end and idx < en for s, en in taken):
            continue  # 与已接受编辑重叠, 丢弃
        taken.append((idx, end))
        located.append((idx, len(e.find), e))
    # 2. 从右到左替换 (位置基于原文, 不漂移)
    out = original
    for idx, length, e in sorted(located, key=lambda t: t[0], reverse=True):
        out = out[:idx] + e.to + out[idx + length:]
    # 3. pause: 在 find 后插入逗号 (在替换结果上重新定位, 片段通常不受 replace 影响)
    for e in edits:
        if e.kind != "pause":
            continue
        idx = -1
        for _ in range(e.occurrence + 1):
            idx = out.find(e.find, idx + 1)
            if idx < 0:
                break
        if idx >= 0:
            end = idx + len(e.find)
            out = out[:end] + "，" + out[end:]
    return out


def _digit_conservation_ok(original: str, result: str) -> bool:
    """数字守恒: 结果的数字集合必须 ⊆ 原文数字集合 (LLM 不得造数字)。"""
    orig_nums = set(_DIGIT_RE.findall(original))
    res_nums = set(_DIGIT_RE.findall(result))
    return res_nums <= orig_nums


# ---------- Agent ----------

_model: OpenAIChatModel | None = None
_disabled: bool = False


def _get_model() -> OpenAIChatModel | None:
    """模型 client 复用 (连接池), Agent 实例每次新建。"""
    global _model, _disabled
    if _model is not None:
        return _model
    if _disabled:
        return None
    api_url = settings.TTS_LLM_TRANSCRIBE_API_URL
    api_key = settings.TTS_LLM_TRANSCRIBE_API_KEY
    model_name = settings.TTS_LLM_TRANSCRIBE_MODEL
    if not (api_url and api_key and model_name):
        _disabled = True
        logger.warning("normalize harness disabled: TTS_LLM_TRANSCRIBE_* not configured")
        return None
    _model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(api_key=api_key, base_url=api_url),
    )
    return _model


def _build_agent(state: HarnessState) -> Agent:
    """每次 run 新建 Agent, 工具闭包捕获本句 state (并发安全)。"""
    model = _get_model()
    agent = Agent(model, name="normalize_editor", system_prompt=SYSTEM_PROMPT)

    @agent.tool_plain
    def submit_edit(find: str, occurrence: int, to: str, rule: str) -> str:
        """提交一条替换编辑。find 必须是原文精确子串 (≥2字符); occurrence 为第几次出现 (0-based); to 为替换文本; rule 为规则类别 (TN/POLY/ABBR/PAUSE/CLEAN)。"""
        _validate_edit(state, find, occurrence, to)
        state.edits.append(Edit("replace", find, occurrence, to, rule))
        return f"OK ({len(state.edits)} edits)"

    @agent.tool_plain
    def submit_pause(after: str, occurrence: int) -> str:
        """在片段 after (原文精确子串) 之后插入停顿逗号。occurrence 0-based。"""
        n = state.original.count(after)
        if n == 0:
            raise ModelRetry(f"after {after!r} 不在原文中。")
        if occurrence < 0 or occurrence >= n:
            raise ModelRetry(f"occurrence={occurrence} 越界 ({after!r} 出现 {n} 次)。")
        state.edits.append(Edit("pause", after, occurrence, "", "PAUSE"))
        return f"OK ({len(state.edits)} edits)"

    @agent.tool_plain
    def query_lexicon(word: str) -> str:
        """查询词表中某词的约定读法。返回 '类别: 读法' 或 '无记录'。"""
        lex = _load_lexicon()
        for category in ("polyphone", "unit", "model"):
            v = (lex.get(category) or {}).get(word)
            if v:
                return f"{category}: {word} → {v}"
        return f"无记录: {word}"

    @agent.tool_plain
    def check_conservation() -> str:
        """自查数字守恒: 预览当前应用结果, 检查数字是否被错误处理。"""
        result = _apply_edits(state.original, state.edits)
        orig_nums = set(_DIGIT_RE.findall(state.original))
        res_nums = set(_DIGIT_RE.findall(result))
        problems = []
        if not res_nums <= orig_nums:
            problems.append(f"结果出现原文没有的数字: {res_nums - orig_nums}")
        if orig_nums and not res_nums and not re.search(r"[零一二三四五六七八九点]", result):
            problems.append("原文数字全部消失且无中文数字")
        if problems:
            return "发现问题: " + "; ".join(problems) + "。请补充编辑修正。"
        return "数字守恒 OK"

    @agent.tool_plain
    def finish() -> str:
        """全部编辑完成, 结束。"""
        return "DONE"

    return agent


async def normalize_sentence(text: str) -> str:
    """单句归一化入口 (与旧 normalizer.normalize_sentence 同签名, 直接替换)。

    失败抛异常, 由调用方 (tts.py endpoint) 决定降级。
    """
    if _get_model() is None:
        return text

    state = HarnessState(original=text)
    agent = _build_agent(state)

    matched = _matched_entries(text)
    lexicon_block = ""
    if matched:
        lexicon_block = (
            "\n\n## 本句命中的词表 (约定读法, 必须遵守)\n" + "\n".join(matched)
        )

    await agent.run(
        f"请归一化这个句子:\n{text}{lexicon_block}",
        model_settings={"thinking": settings.TTS_LLM_REASONING_EFFORT},
    )

    if not state.edits:
        return text  # 模型判断无需编辑

    final = _apply_edits(text, state.edits)
    if not _digit_conservation_ok(text, final):
        logger.warning(f"harness digit conservation failed, fallback original: {text[:50]}")
        return text
    return final
