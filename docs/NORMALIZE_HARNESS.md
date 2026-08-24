# TTS Normalize Harness 实现文档

> 位置: `backend/app/services/normalize_harness.py` + `tts_lexicon.yaml`
> 接口: `POST /api/v1/tts/normalize` (单句) / `POST /api/v1/tts/normalize-batch` (并发批)
> 上游: md-reader 前端逐句预取 (`/api/tts/normalize` proxy)

## 1. 是什么

把原始文本转成 TTS 引擎能正确朗读的文本。**agent 方式**: 模型不重写全文, 只提交编辑指令, 程序负责应用与校验——错误在源头被拦截, 不会污染输出。

```
原句 → 词表命中扫描 → 动态注入 prompt
     → pydantic-ai Agent (每次 run 新建, thinking 可配)
     → 模型自主多轮调用工具 (3+ requests 典型)
     → 每个工具调用即时校验, 不合格 raise ModelRetry 回喂模型修正
     → 程序从右到左应用 edits → 数字守恒终检 → tts_text
```

## 2. 为什么是 agent 而不是单次调用

历史演进 (三版):

| 版本 | 形式 | 结果 |
|---|---|---|
| v1 自由文本 | LLM 直接输出改写全文 | 882 处拼音过度标注 / 大段原样返回 / 丢数字 |
| v2 单次 edits JSON | `output_type=BaseModel` 结构化输出 | DeepSeek 拒绝: tool_choice=required 与 thinking 互斥, 全档位 400 |
| v3 agent 工具 | 工具注册 (tool_choice=auto) + ModelRetry | 0 原样返回 / 4 处合法拼音 / 0 数字丢失 |

关键实测结论 (DeepSeek 能力矩阵):
- `json_object` + thinking: 可用 (enabled 慢 4x)
- `json_schema` response_format: 400 不支持
- `tool_choice=required` (pydantic-ai output_type 内部机制): 全档位 400
- **`tool_choice=auto` (工具注册) + thinking: 可用** ← 本方案
- thinking=enabled 实测不慢反快 (16.3s vs 20.6s)

agent 方式的本质优势: **校验时机**。单次调用是"全部返回后一次性校验, 错了只能丢弃"; agent 是"每个编辑提交时立即校验, 错误以 ModelRetry 回喂, 模型当场修正"——harness 反馈环。

## 3. 工具集 (5 个)

工具在 `_build_agent(state)` 里以闭包捕获该句的 `HarnessState`, **每次 run 新建 Agent 实例** (并发安全, 不在全局单例上累积注册):

| 工具 | 签名 | 作用 |
|---|---|---|
| `submit_edit` | `(find, occurrence, to, rule)` → `OK (N edits)` | 替换编辑。find 须原文精确子串 ≥2 字符 |
| `submit_pause` | `(after, occurrence)` → `OK (N edits)` | 在片段后插入停顿逗号 |
| `query_lexicon` | `(word)` → `类别: 读法` / `无记录` | 查词表约定读法 (词表最高优先级) |
| `check_conservation` | `()` → `数字守恒 OK` / `发现问题: ...` | 模型自查数字守恒 |
| `finish` | `()` → `DONE` | 结束 |

`occurrence` 语义: 同一片段多次出现时用第几次 (0-based) 定位。解决"多重压力...多重选择"同句两词不同读音的定位问题。

## 4. harness 校验 (程序侧)

### 4.1 工具内即时校验 (`_validate_edit`)

| 规则 | 检查 | 失败动作 |
|---|---|---|
| find 存在 | `original.count(find) > 0` | ModelRetry: "find 不在原文中" |
| occurrence 界 | `0 <= occurrence < count` | ModelRetry: "occurrence 越界" |
| find 长度 | `len(find) >= 2` | ModelRetry: "太短无法定位" |
| 虚词黑名单 | find ∉ {的,了,着,地,得,和,还,都,从,同,当,...} | ModelRetry: "虚词禁止标音" (治 v1 的 de 灾难) |
| 拼音内禁空格 | to 不匹配 `拼音 拼音` 模式 | ModelRetry (治 "gōng yìng链" 被拆读) |
| to 有效性 | 含中文/字母/常用标点 | ModelRetry (拦纯异常; **符号→符号替换合法**——破折号→逗号是清洗操作) |

ModelRetry 被拒绝后 pydantic-ai 自动把错误信息回喂模型, 模型修正重提 (日志实测: 破折号纯符号替换被拒 → 模型改用带上下文的 find " — "第一梯队"" → 通过)。

### 4.2 应用算法 (`_apply_edits`)

1. replace 全部**基于原文定位** (不在中间结果上 find, 防漂移)
2. 重叠检测: 与已接受编辑区间重叠的后提交者丢弃 (治 DRAM → "D R A MA M" 碎片)
3. 从右到左替换 (位置不漂移)
4. pause 在 replace 之后于结果上定位插入逗号

### 4.3 终检降级

- `_digit_conservation_ok`: 结果数字集合 ⊆ 原文数字集合 (LLM 不得造数字), 失败 → 整句降级原文
- `edits` 为空 → 返回原文 (模型判断无需处理)
- 任何异常 → tts.py 端点 catch → 降级原文。**永不输出比原文更糟的文本**

## 5. 词表 (`tts_lexicon.yaml`)

三类, **命中才注入** prompt (动态 few-shot, 没命中就没有"的→de"灾难):

```yaml
polyphone:   # 多音字 — 只收 TTS 上下文消歧失败的高危词 (18 条)
  参差: 参cēn差
  累计: 累lěi计
unit:        # 单位/缩写 (45 条)
  GB: G B
  GHz: G 赫兹
model:       # 型号 (28 条)
  GLM-5.3: G L M 五点三
  YOLO: YOLO        # 缩读词不拆
  YOLO11: YOLO 十一
```

维护方式: 发现一个错例加一条。词表是最高优先级 (模型必须遵守注入条目)。

**polyphone 收录红线**: 上下文能读对的常见多音词 (银行/音乐/成长/处理/重要/重新...) 一律不收——标注反而引入双读风险 (v1 教训: 拼音紧跟汉字的形式如果 TTS 拆开读就是双音节)。只有 TTS 实际读错且无法靠上下文的才进表。

## 6. prompt 规则要点 (5 节)

1. **数字转中文** (TTS 读不了 ASCII 数字/小数点): 年份逐字 / 数量按位 / 2+量词→两 / 小数版本号必转 (5.3→五点三) / 长ID空格逐字 1 读幺
2. **英文二分法**: 缩读词 acronym (YOLO/NASA/Astra, 元辅音交替当词读) 保持原样; 首字母缩写 initialism (CPU/GPU/DRAM/ETF, 辅音连串逐字母) 拆开空格; 词表优先
3. **多音字保守**: 常见词不处理靠上下文; 高危词查词表; 拼音紧跟字无括号 (重新→chóng新); 虚词禁标
4. **停顿语义边界**: 状语后/主谓间/并列间; "英文+助词(里/的/中)"停顿放助词后 ("邀请 Gemini 里，想离开"); 动宾短语内部禁拆 (想离开/要做/能吃)
5. **符号清洗**: markdown 去除 / 破折号→逗号 / URL 保留

## 7. 并发与性能

- `/normalize-batch`: `asyncio.gather` + `asyncio.Semaphore(8)` (tts.py:143)
- harness 每次 run 新建 Agent, state 闭包独立 → 并发安全
- 模型 client (`OpenAIChatModel`) 全局复用 (连接池)
- 实测: 41 段 29KB 文章 535s (~13s/句, thinking=low); 前端 5 并发预取流水线下不阻塞播放

## 8. 配置

```env
TTS_LLM_TRANSCRIBE_ENABLED=true
TTS_LLM_TRANSCRIBE_API_URL=https://api.deepseek.com
TTS_LLM_TRANSCRIBE_API_KEY=sk-...
TTS_LLM_TRANSCRIBE_MODEL=deepseek-v4-flash
TTS_LLM_REASONING_EFFORT=low      # thinking 档位, 传给 model_settings
```

## 9. 回归验证方法

测试集: `note/twitter-daily/<date>/brief/brief.md` (真实策展简报, 最难的场景: 数字密度高/型号多/中英混排)

```bash
/mnt/media3/project/edge-tts/venv-asr/bin/python \
  /mnt/media3/project/edge-tts/tmp/norm_brief.py <brief.md> <out.txt>
# 输出 ORIG/NORM 对照表, 统计: 原样返回段数 / 拼音标注数 / 数字残留
```

质量基线 (2026-08-06 brief, 41 段):
- 原样返回 0/41 (纯标题段合理)
- 拼音标注 4 处 (全部词表内)
- 数字残留 0 (500亿元→五百亿元 / 1/10→四十六分之一 / 0.3%→百分之零点三)

读音验证 (TTS→ASR 回环): 见 `research/qwen3-asr.cpp/USAGE.md` (注意 ASR 有语义纠错, 判双读必须用音频时长对照)。
