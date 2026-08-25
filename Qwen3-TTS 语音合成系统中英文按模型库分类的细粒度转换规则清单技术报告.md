# Qwen3-TTS 语音合成系统中英文按模型库分类的细粒度转换规则清单技术报告

---

## 1\. 报告说明与架构边界

本报告针对**中文与英文（Multilingual/English）两条路线，不做任何主观混淆或抽象简写，完全按不同的开源模型库/前端框架独立拆解，逐一列出每一个模型库内部所实现的具体英文与中文文本转换规则、输入 \-\> 输出映射关系、控制标记及对应的源码文件/函数引用**。

---

## 2\. 工业级开源文本预处理库转换清单 (中文与英文)

### 2.1 WeTextProcessing (WeNet / 出门问问开源)

* **官方仓库与文档**: [wenet-e2e/WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)  
* **核心模块路径**: `tn/chinese/rules/*.py` (中文 TN) 与 `tn/english/rules/*.py` (英文 TN)

#### A. 中文细粒度规则清单 (`tn/chinese/rules/*.py`)：

| 规则分类 | 源码对应文件 | 原始输入文本 (Raw Text) | 转换后文本 (Normalized Text) | 规则逻辑与语义说明 |
| :---- | :---- | :---- | :---- | :---- |
| **年份转写** | `date.py` | `2026年8月7日` | `二零二六年八月七日` | 4 位数字跟“年”时，强制逐字转中文数字，不按千百十念 |
| **基数/数量** | `cardinal.py` | `2026个` | `两千零二十六个` | 数字跟通用量词时，转为千/百/十读法 |
| **量词“2”** | `cardinal.py` | `2个` | `两个` | 单独数字 `2` 跟量词时由“二”转写为“两” |
| **时间转写** | `time.py` | `14:30` | `十四点三十分` | 冒号时间格式转为点分读法 |
| **分数/百分比** | `fraction.py`/`percentage.py` | `1/5`, `6.3%` | `五分之一`, `百分之六点三` | 符号显式汉字化展开 |
| **电话/ID** | `telephone.py` | `13800000000` | `一 三 八 零 零 零 零 零 零 零 零` | 5 位以上纯数字空两格隔开，数字`1`读`幺` |

#### B. 英文细粒度规则清单 (`tn/english/rules/*.py`)：

| 规则分类 | 源码对应文件 | 原始输入文本 (Raw Text) | 转换后文本 (Normalized Text) | 规则逻辑与语义说明 |
| :---- | :---- | :---- | :---- | :---- |
| **基数词 (Cardinal)** | `cardinal.py` | `123 apples` | `one hundred twenty-three apples` | 基础阿拉伯数字转为标准英文基数词 |
| **序数词 (Ordinal)** | `ordinal.py` | `1st, 2nd, 22nd` | `first, second, twenty-second` | 带有缩写后缀的序数词展开 |
| **日期转写 (Date)** | `date.py` | `2026-08-07` | `August seventh twenty twenty-six` | 日期格式按英文习惯展开为月份+序数词+年份 |
| **时间转写 (Time)** | `time.py` | `8:30 AM` | `eight thirty a m` | 英文时间与 AM/PM 展开为字母拼读 |
| **小数转写 (Decimal)** | `decimal.py` | `3.14` | `three point one four` | 小数点 `.` 转写为 `point` |
| **分数转写 (Fraction)** | `fraction.py` | `1/2, 3/4` | `one half, three fourths` | 斜杠分数转写为基数词+序数词复数 |
| **货币转写 (Money)** | `money.py` | `$100, €50` | `one hundred dollars, fifty euros` | 前置货币符号移至尾部并展开全称 |
| **度量衡 (Measure)** | `measure.py` | `10kg, 60mph` | `ten kilograms, sixty miles per hour` | 缩写单位转为英文法定度量衡 |
| **通用缩写 (Abbreviation)** | `abbreviation.py` | `Dr. Smith, St. John` | `Doctor Smith, Saint John` | 尊称与地名缩写补全 |
| **应用缩写 (Acronym)** | `abbreviation.py` | `e.g. cats, i.e. that` | `for example cats, that is that` | 常用拉丁缩写展开为英文全称 |
| **罗马数字 (Roman)** | `roman.py` | `Chapter III, Henry VIII` | `Chapter three, Henry the eighth` | 根据上下文环境判断为基数词或序数词 |

---

### 2.2 pypinyin & g2p\_en (中文与英文 G2P 库)

* **官方仓库与文档**: [python-pinyin](https://github.com/mozillazg/python-pinyin)，[g2p\_en (Kyubyong)](https://github.com/kyubyong/g2p)  
* **核心源码文件**: `pypinyin/phrases_dict.py`, `g2p_en/g2p.py`

#### 细粒度多音字与英文异读词 (Heteronyms) 转换清单：

| 语言类型 | 控制模块/接口 | 原始输入文本示例 | 处理后音素/文本 | 规则逻辑与消歧机理 |
| :---- | :---- | :---- | :---- | :---- |
| **中文多音字** | `pypinyin.phrases_dict` | `“重新开始”` | `chóng xīn kāi shǐ` | 词组匹配词典消歧，阻止错读为 `zhòng` |
| **中文自定义** | `load_phrases_dict()` | `“单于”` | `chán yú` | 用户自定义词典覆盖默认规则 |
| **英文异读词** | `g2p_en` (POS 词性标注) | `“I read books”` | `[R, IY1, D]` (read /riːd/) | 动词现在时词性解析发长音 /iː/ |
| **英文异读词** | `g2p_en` (POS 词性标注) | `“I read a book yesterday”` | `[R, EH1, D]` (read /rɛd/) | 动词过去时词性解析发短音 /ɛ/ |
| **英文异读词** | `g2p_en` (POS 词性标注) | `“lead pipe”` vs `“lead the team”` | `[L, EH1, D]` vs `[L, IY1, D]` | 名词(金属铅 \[lɛd\]) 与 动词(引导 \[liːd\]) 词性消歧 |
| **英文异读词** | `g2p_en` (POS 词性标注) | `“live in NY”` vs `“live music”` | `[L, IH1, V]` vs `[L, AY1, V]` | 动词(居住 \[lɪv\]) 与 形容词/现场(\[laɪv\]) 消歧 |
| **英文异读词** | `g2p_en` (POS 词性标注) | `“I refuse to collect the refuse”` | `[R, IH0, F, Y, UW1, Z]` vs `[R, EH1, F, Y, UW2, S]` | 动词(拒绝) 与 名词(垃圾) 声调与重音消歧 |

---

## 3\. 开源 TTS 框架前端模块英文转换清单

### 3.1 CosyVoice.TextFrontend (阿里巴巴 FunAudioLLM)

* **源码链接**: [GLM-TTS/cosyvoice/cli/frontend.py](https://github.com/zai-org/GLM-TTS/blob/main/cosyvoice/cli/frontend.py)  
* **核心英文函数**: `TextFrontend._normalize_english_text()`

#### 细粒度英文预处理转换清单：

| 转换模块/步骤 | 源码映射位置 (`frontend.py`) | 原始输入文本示例 | 处理后输出文本 | 转换逻辑与作用 |
| :---- | :---- | :---- | :---- | :---- |
| **英文数字拼读** | `spell_out_number()` | `100 apples, in 2026` | `one hundred apples, in twenty twenty-six` | 调用 `inflect` 库将纯数字展开为英文单词 |
| **英文算式符号** | `replace_asterisk_with_multiply()` | `2*3` | `two times three` | 将星号 `*` 转写为 `times` |
| **英文缩写展开** | `contractions.fix()` | `don't, I'm, it's, won't` | `do not, I am, it is, will not` | 将英文否定/代词缩写展开为完整动词格式 |
| **非法括号清洗** | `remove_bracket('en')` | `Hello (world) [TTS]` | `Hello world` | 剥离非朗读性注释标记 |
| **长破折号平滑** | `text.replace('—', ' ')` | `well-known — indeed` | `well-known   indeed` | 将英文破折号转为空格，防断音混淆 |

---

## 4\. ComfyUI 生态 Qwen3-TTS 节点中英文转换清单 (ComfyUI-Qwen-TTS)

### 4.1 节点中英文参数与拼读规则全集

* **源码链接**: [flybirdxx/ComfyUI-Qwen-TTS/nodes.py](https://github.com/flybirdxx/ComfyUI-Qwen-TTS/blob/main/nodes.py)

#### 细粒度中英文参数控制清单：

| 节点参数 / 文本格式 | 语言类型 | 输入文本 / 选项示例 | 对应生成效果与机制 |
| :---- | :---- | :---- | :---- |
| **英文全大写空格** | 英文 | `A P P`, `C E O`, `U S A` | 强制切分为单字母 Token，触发逐字拼读 (`A-P-P`) |
| **英文小写/驼峰** | 英文 | `App`, `ChatGPT`, `Nasa` | 模型识别为名词单词，触发单词连读 (\[æp\]) |
| **`language` 参数** | 多语言 | `English`, `Chinese`, `Japanese` | 切换语音编码器的语言先验偏置，防口音漂移 |
| **`instruct` 口音 Prompt** | 英文 | `Instruction: British male accent, crisp RP` | 在 `VoiceDesign` / `CustomVoice` 中注入标准英音/美音口音 |
| **`instruct` 语速 Prompt** | 英文 | `Instruction: Fast-paced delivery, rapid rate` | 在英文朗读中加速自回归 Token 步进 |

---

## 5\. 参考文献与数据源声明

1. [**WeTextProcessing 英文 TN 源码集 (`tn/english/rules/*.py`)**](https://github.com/wenet-e2e/WeTextProcessing) — 工业级英文文本正则化规则源文件。  
2. [**CosyVoice 英文预处理源码 (`cosyvoice/cli/frontend.py`)**](https://github.com/zai-org/GLM-TTS/blob/main/cosyvoice/cli/frontend.py) — 英文缩写展开与数字拼读源码。  
3. [**g2p\_en 英文 G2P 与异读词词性消歧源码 (`g2p_en/g2p.py`)**](https://github.com/kyubyong/g2p) — 英文异读词消歧规则。  
4. [**Qwen3-TTS 官方 README 与多语言说明**](https://github.com/QwenLM/Qwen3-TTS/blob/main/README.md) — 英文与多语言生成规范。

