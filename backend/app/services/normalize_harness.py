"""TTS 文本归一化 agent harness。

架构 (agent 方式, 非单次调用):
    原句 → 词表命中扫描 → 动态注入 prompt
         → pydantic-ai Agent (每次 run 新建实例, 工具闭包捕获该句 state)
         → 模型自主多轮调用工具:
             submit_edit(find, occurrence, to, rule)  替换编辑
             submit_pause(after, occurrence, kind)    插入停顿 (kind: space 空格 / comma 逗号)
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

# 确定性规则: 字母-2-字母 → 字母 to 字母 (B2B → B to B, P2P → P to P)
# 捕获组保证 B2B2C → B to B to C (链式); M2 芯片 不误伤 (2 后无字母)
_DIGIT_TO_RE = re.compile(r"([A-Za-z])2([A-Za-z])")


def _deterministic_preprocess(text: str) -> str:
    """程序侧确定性归一化 (不进 LLM): X2X 模式的 2 读 'to'。

    B2B/B2C/P2P/C2C/A2A/G2B/O2O/M2M 等商业/技术术语的 2 都读 'to' (双),
    这是纯规则, 由程序直接处理, 避免每句让 LLM 猜。
    """
    return _DIGIT_TO_RE.sub(r"\1 to \2", text)


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

### 3. 多音字 (pypinyin 扫描表驱动, 必须逐一判断)
- 的/地/得 作结构助词时读轻声 de, TTS 总读错, 统一替换为同音汉字"的" (不加拼音, 无声调拼音会被当字母双读):
  我的书 → 我 的 书; 隐晦地 → 隐晦 的; 跑得快 → 跑 的 快
  即: 助词"地"→"的", 助词"得"→"的", "的"本身保留
  的地得其他用法不替换: 目的dì/的确dí/得手dé
- 程序已用 pypinyin 扫出本句全部多音字及候选读音 (见下方"多音字扫描表")
- **对扫描表中每个字, 逐一结合语境判断读音**:
  - 常见多音词语境唯一 (银行/音乐/成长/处理/重要/重新) → TTS 自己会读对, 跳过不标
  - 语境有歧义、TTS 可能读错 → 必须用 submit_edit 的 pinyin 参数标注
  - 候选音都不合适时, 也可提交候选外的读音 (程序会校验该字确有此音)
- 扫描表之外的 (词级多音词/罕见字), 你认为 TTS 会读错的也可以标注
- **多音字标注格式 (重要)**: 用 submit_edit 的 pinyin 参数给出读音, 程序会校验合法性并规范化格式:
  - pinyin 可写 chēng / cheng1 / cheng 任意格式 (程序统一转带调号 chēng)
  - to 里写替换后的完整文本 (拼音完全替换被标字, 紧跟前后字): 重新 → chóng新
  - 不要自己写括号/数字声调/声字 (cheng4声 是错的)
  - 程序会检测读音对该字是否合法, 不合法会拒绝并告诉你合法读音
- 标音前必须确认正确读音 (用 query_lexicon 或词典), 标错比不标更糟
- 严禁给 从/同/当/了/着 等虚词标音 (语境绝对唯一, 标注反而引入双读)

### 4. 停顿 (语义单元粒度, 考虑位置与时长)
- 停顿粒度 = 语义单元 (不只是长句才分): 每个意群之间都要有分界
  例: 苹果 起诉 爆料人 Jon Prosser 的 诉讼 在 取证环节 出现 拖延
  (苹果-起诉-爆料人Jon Prosser的-诉讼-在-取证环节-出现拖延, 每个主谓宾块都分)
- 停顿只放在语义单元边界: 状语之后、主谓之间、并列成分之间
- 英文词/名字 + 中文助词 (里/的/中/上) 结尾的状语, 停顿放助词之后:
  "邀请 Gemini 里 想离开的人" ✓ (Gemini 里 = 状语, 想离开 = 新谓语)
  "邀请 Gemini 里想 离开的人" ✗ (拆散了动宾短语 想离开)
- 动宾短语/谓语内部禁止停顿: 想离开/要做/能吃/会来 不能拆
- **用 submit_pause 提交停顿, 必须指定 kind (停顿时长三档, 依次递增)**:
  - kind="space": 微停顿 (分词粒度, 句中主谓宾块之间) — 绝大多数停顿用它
  - kind="comma": 短停顿 (意群/分句之间; 长句 20 字以上必须有)
  - kind="semicolon": 中停顿 (大意群转折/强调分隔, 少用)
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
    pause_kind: str = "space"  # pause 时: "space" 微停顿 | "comma" 短停顿 | "semicolon" 中停顿


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


def _scan_polyphone_hints(text: str) -> list[str]:
    """pypinyin 预扫描: 找出句中全部多音字及候选读音 (注入 prompt 供 LLM 选音)。

    pypinyin 是信息提供者 (事前), 不是校验器 (事后):
    - 扫出的多音字 + 候选音 = 必须逐一判断的内容
    - LLM 从候选选音; 也可选候选外的合法音; 也可处理扫描外的字
    """
    from pypinyin import pinyin as pypinyin_all, Style

    lines: list[str] = []
    seen: set[str] = set()
    for i, ch in enumerate(text):
        if not ("\u4e00" <= ch <= "\u9fff") or ch in seen:
            continue
        cand = sorted({p for pl in pypinyin_all(ch, style=Style.TONE, heteronym=True) for p in pl})
        if len(cand) > 1:
            seen.add(ch)
            ctx = text[max(0, i - 4) : i] + "「" + ch + "」" + text[i + 1 : i + 5]
            lines.append(f"  {ch} (候选: {'/'.join(cand)}) 出现于 …{ctx}…")
    return lines


# ---------- harness 校验 (工具内联) ----------

# 拼音声调: 数字声调 → 调号符号 (cheng1 → chēng)
# 调号顺序与 _VOWEL_ORDER (a o e i u ü) 对齐
_VOWEL_ORDER = "aoeiuv"  # v = ü
_TONE_MARKS = {
    "1": "āōēīūǖ",
    "2": "áóéíúǘ",
    "3": "ǎǒěǐǔǚ",
    "4": "àòèìùǜ",
}
_TONE_STRIP = str.maketrans("āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜ", "aaaaoooeeeeeiiiiuuuuvvvv")


def _normalize_tone(pinyin: str) -> str:
    """把各种声调格式统一为带调号拼音: cheng1→chēng, cheng→cheng(无调), chēng 保持。

    规则: 声调数字跟在韵母元音后 → 移到韵腹元音上加调号 (普通话标调规则:
    a 优先, 其次 o/e, 再 i/u/ü); 无声调数字保持。
    """
    if not pinyin:
        return pinyin
    m = re.match(r"^([a-zA-ZüÜ]+)([1-4])$", pinyin)
    if not m:
        return pinyin.lower()  # 已是带调号或纯拼音, 统一小写
    base, tone = m.group(1).lower(), int(m.group(2))
    base = base.replace("ü", "v")  # 统一用 v 处理韵腹
    # 找韵腹 (a 优先, o/e 次之, i/u/ü 最后)
    target = None
    for ch in "aoeiuv":
        if ch in base:
            target = ch
            break
    if target is None:
        return base.replace("v", "ü") + str(tone)
    idx = base.index(target)
    mark = _TONE_MARKS[str(tone)][_VOWEL_ORDER.index(target)]
    return (base[:idx] + mark + base[idx + 1:]).replace("v", "ü")


def _pinyin_legal(chars: str, pinyin: str) -> tuple[bool, set[str]]:
    """用 pypinyin 查 chars 里每个字的合法读音, 验证 pinyin (无调) 是否在候选里。

    返回 (是否合法, 合法读音集合 (带调, 如 {chēng, chèn}) )。
    """
    from pypinyin import pinyin as pypinyin_all, Style

    legal: set[str] = set()
    for ch in chars:
        if "\u4e00" <= ch <= "\u9fff":
            # heteronym=True 返回全部读音 (多音字: 称 chēng/chèn/chèng)
            for plist in pypinyin_all(ch, style=Style.TONE, heteronym=True):
                for p in plist:
                    legal.add(p)
    target = _normalize_tone(pinyin)  # 转带调
    target_plain = target.translate(_TONE_STRIP)
    for p in legal:
        if p.translate(_TONE_STRIP) == target_plain:
            return True, legal
    return False, legal


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
    # 3. pause: 在 find 后插入空格或逗号 (在替换结果上重新定位, 片段通常不受 replace 影响)
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
            sep = {"space": " ", "comma": "，", "semicolon": "；"}.get(e.pause_kind, " ")
            out = out[:end] + sep + out[end:]
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
    def submit_edit(find: str, occurrence: int, to: str, rule: str, pinyin: str = "") -> str:
        """提交一条替换编辑。find 必须是原文精确子串 (≥2字符); occurrence 为第几次出现 (0-based); to 为替换文本; rule 为规则类别 (TN/POLY/ABBR/PAUSE/CLEAN); pinyin 可选 — 多音字标注时给出读音 (如 chēng/cheng1/cheng), 程序校验并规范格式。"""
        _validate_edit(state, find, occurrence, to)
        if pinyin:
            ok, legal = _pinyin_legal(find, pinyin)
            if not ok:
                legal_str = "/".join(sorted(legal)) if legal else "(查无)"
                raise ModelRetry(
                    f"读音 {pinyin!r} 对 {find!r} 不合法。该字合法读音: {legal_str}。请修正 pinyin 后重提。"
                )
            # 把 to 里的拼音 (可能是数字声调/无调) 规范化为带调号
            norm = _normalize_tone(pinyin)
            to = re.sub(r"[a-zA-ZüÜāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]+", norm, to, count=1)
        state.edits.append(Edit("replace", find, occurrence, to, rule))
        return f"OK ({len(state.edits)} edits)"

    @agent.tool_plain
    def submit_pause(after: str, occurrence: int, kind: str = "space") -> str:
        """在片段 after (原文精确子串) 之后插入停顿。kind (时长递增): "space"=微停顿(分词), "comma"=短停顿(意群), "semicolon"=中停顿(大意群分隔)。occurrence 0-based。"""
        n = state.original.count(after)
        if n == 0:
            raise ModelRetry(f"after {after!r} 不在原文中。")
        if occurrence < 0 or occurrence >= n:
            raise ModelRetry(f"occurrence={occurrence} 越界 ({after!r} 出现 {n} 次)。")
        if kind not in ("space", "comma", "semicolon"):
            raise ModelRetry(f"kind={kind!r} 无效, 只允许 'space'/'comma'/'semicolon'。")
        state.edits.append(Edit("pause", after, occurrence, "", "PAUSE", kind))
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

    text = _deterministic_preprocess(text)  # 程序侧规则先跑: B2B → B to B
    state = HarnessState(original=text)
    agent = _build_agent(state)

    matched = _matched_entries(text)
    lexicon_block = ""
    if matched:
        lexicon_block = (
            "\n\n## 本句命中的词表 (约定读法, 必须遵守)\n" + "\n".join(matched)
        )

    poly_hints = _scan_polyphone_hints(text)
    poly_block = ""
    if poly_hints:
        poly_block = (
            "\n\n## 多音字扫描表 (pypinyin 预扫描, 每个字都必须逐一判断读音)\n"
            + "\n".join(poly_hints)
        )

    await agent.run(
        f"请归一化这个句子:\n{text}{lexicon_block}{poly_block}",
        model_settings={"thinking": settings.TTS_LLM_REASONING_EFFORT},
    )

    if not state.edits:
        return text  # 模型判断无需编辑

    final = _apply_edits(text, state.edits)
    if not _digit_conservation_ok(text, final):
        logger.warning(f"harness digit conservation failed, fallback original: {text[:50]}")
        return text
    return re.sub(r" {2,}", " ", final)  # 压缩连续空格 (停顿插在已有空格后)
