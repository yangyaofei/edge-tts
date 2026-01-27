# Qwen3-TTS FastAPI 集成使用指南

> **更新时间**: 2026-01-26

本指南说明如何使用已集成到 FastAPI 服务中的 Qwen3-TTS 功能。

---

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

这将安装：
- `qwen-tts`: Qwen3-TTS Python 包
- `torch`: PyTorch (支持 CUDA 和 MPS)
- `soundfile`: 音频处理
- `numpy`: 数值计算

### 2. 配置环境变量

复制示例配置文件：

```bash
mkdir -p config
cp backend/.env.example config/config.env
```

编辑 `config/config.env`，配置 Qwen3-TTS：

```env
# 启用 Qwen3-TTS
QWEN_ENABLE=true

# 模型配置
QWEN_MODEL_TYPE=CustomVoice  # CustomVoice, Base, VoiceDesign
QWEN_MODEL_SIZE=1.7B         # 1.7B (高质量) 或 0.6B (轻量级)
QWEN_DEVICE=                 # 留空=自动检测 (CUDA>MPS>CPU)
```

### 3. 启动服务

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

首次启动会：
1. 自动生成 JWT Token（打印在控制台）
2. 下载 Qwen3-TTS 模型（~3.5GB，首次较慢）
3. 初始化模型到 GPU/CPU

看到以下输出表示成功：

```
============================================================
✓ Qwen3-TTS initialized successfully!
  Model: Qwen3-TTS-12Hz-1.7B-CustomVoice
  Device: cuda:0
============================================================
```

---

## API 使用方法

### 方式一：使用统一的流式端点

**端点**: `POST /api/v1/tts/stream`

```bash
curl -X POST "http://localhost:8000/api/v1/tts/stream" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是测试语音。",
    "engine": "qwen_tts",
    "speaker": "Vivian",
    "language": "Chinese"
  }' \
  --output test.wav
```

### 方式二：使用 Qwen 专用端点

#### CustomVoice 模式（推荐）

```bash
curl -X POST "http://localhost:8000/api/v1/tts/qwen_tts/generate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是 Qwen3-TTS 生成的语音。",
    "speaker": "Vivian",
    "language": "Chinese",
    "instruct": "用温柔的声音说"
  }' \
  --output test.wav
```

#### VoiceDesign 模式

```bash
curl -X POST "http://localhost:8000/api/v1/tts/qwen_tts/design" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello! This is a test.",
    "instruct": "Speak in a cheerful and energetic tone",
    "language": "English"
  }' \
  --output test.wav
```

#### VoiceClone 模式

```bash
curl -X POST "http://localhost:8000/api/v1/tts/qwen_tts/clone" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是克隆的声音在说话。",
    "ref_audio_url": "/path/to/reference.wav",
    "ref_text": "参考音频的文字内容",
    "language": "Chinese"
  }' \
  --output test.wav
```

---

## Python 客户端示例

```python
import requests

# 配置
BASE_URL = "http://localhost:8000"
TOKEN = "your_admin_token_here"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# 生成语音
response = requests.post(
    f"{BASE_URL}/api/v1/tts/qwen_tts/generate",
    headers=headers,
    json={
        "text": "你好，这是 Qwen3-TTS 生成的语音。",
        "speaker": "Vivian",
        "language": "Chinese",
        "instruct": "用温柔的声音说"
    }
)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("✓ 音频已保存到 output.wav")
else:
    print(f"✗ 错误: {response.status_code} - {response.text}")
```

---

## 支持的说话人

| 说话人 | 声音描述 | 原生语言 | 推荐场景 |
|-------|---------|---------|---------|
| Vivian | 明亮、略带棱角的年轻女性声音 | 中文 | 通用女声 |
| Serena | 温柔、温和的年轻女性声音 | 中文 | 温馨 narration |
| Uncle_Fu | 成熟、低沉醇厚的男声 | 中文 | 正式内容 |
| Dylan | 清晰自然的北京腔年轻男声 | 中文（北京方言） | 亲切对话 |
| Eric | 活泼、略带沙哑明亮的成都男声 | 中文（四川方言） | 活泼内容 |
| Ryan | 动感十足、节奏感强的男声 | 英语 | 体育、新闻 |
| Aiden | 阳光开朗的美国男声 | 英语 | 友好对话 |
| Ono_Anna | 顽皮轻盈的日本女性声音 | 日语 | 可爱内容 |
| Sohee | 温暖情感丰富的韩国女性声音 | 韩语 | 情感表达 |

---

## 支持的语言

- `Auto`: 自动检测
- `Chinese`: 中文
- `English`: 英语
- `Japanese`: 日语
- `Korean`: 韩语
- `French`: 法语
- `German`: 德语
- `Spanish`: 西班牙语
- `Portuguese`: 葡萄牙语
- `Russian`: 俄语

---

## 设备配置

### 自动检测（推荐）

```env
QWEN_DEVICE=
```

系统会自动选择：CUDA > MPS > CPU

### 强制使用 CUDA

```env
QWEN_DEVICE=cuda:0
```

### 强制使用 MPS (Apple Silicon)

```env
QWEN_DEVICE=mps
```

### 强制使用 CPU

```env
QWEN_DEVICE=cpu
```

---

## 性能优化建议

### NVIDIA GPU (CUDA)

1. **安装 CUDA 版 PyTorch**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

2. **启用 FlashAttention 2**（可选，减少显存）:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

3. **使用 bfloat16**（自动）:
   ```env
   # CUDA 自动使用 bfloat16 获得更好性能
   ```

### Apple Silicon (MPS)

1. **使用 MPS 后端**:
   ```env
   QWEN_DEVICE=mps
   ```

2. **使用 float16**（自动）:
   ```python
   # MPS 自动使用 float16（更稳定）
   ```

3. **考虑 MLX 框架**（未来）:
   - MLX 是 Apple 原生框架，性能更好
   - 目前版本使用 PyTorch MPS

### CPU (不推荐)

```env
QWEN_DEVICE=cpu
```

- 速度很慢（可能 10-30x）
- 仅用于测试或没有 GPU 的环境

---

## 常见问题

### Q1: 模型下载太慢或失败？

**解决方案**:

1. **使用 Hugging Face 镜像**（中国大陆）:
   ```env
   HF_ENDPOINT=https://hf-mirror.com
   ```

2. **手动下载模型**:
   ```bash
   pip install modelscope
   modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./models/qwen-tts
   ```

3. **设置环境变量**:
   ```bash
   export HF_HUB_ENABLE_HF_TRANSFER=1
   ```

### Q2: 显存不足？

**解决方案**:

1. **使用更小的模型**:
   ```env
   QWEN_MODEL_SIZE=0.6B
   ```

2. **使用 CPU**（速度慢但无显存限制）:
   ```env
   QWEN_DEVICE=cpu
   ```

3. **关闭其他占用 GPU 的程序**

### Q3: MPS 上的错误？

**已知限制**:

- 部分算子可能不支持，会回退到 CPU
- `bfloat16` 支持不完善，自动使用 `float16`

**解决方案**:

```env
QWEN_DEVICE=cpu  # 如果 MPS 有问题，改用 CPU
```

### Q4: 如何在浏览器中测试？

1. 打开 Swagger UI:
   ```
   http://localhost:8000/docs
   ```

2. 找到 `/api/v1/tts/qwen_tts/generate` 端点

3. 点击 "Try it out"

4. 填写参数并执行

5. 下载生成的音频

---

## 运行测试

使用提供的测试脚本：

```bash
cd backend

# 设置 token（可选，localhost 可能不需要）
export TTS_TOKEN="your_admin_token"

# 运行测试
python test_qwen_tts.py
```

测试会：
1. 检查服务健康状态
2. 获取可用语音列表
3. 测试基本语音生成
4. 测试多语言支持
5. 测试不同说话人
6. 测试风格指令
7. 测试统一端点
8. 测试长文本处理

生成的音频文件会保存在当前目录（`test_*.wav`）

---

## API 端点总览

| 端点 | 方法 | 描述 |
|------|-----|------|
| `/api/v1/tts/voices?engine=qwen_tts` | GET | 获取语音列表 |
| `/api/v1/tts/health/qwen` | GET | Qwen 模型健康检查 |
| `/api/v1/tts/qwen_tts/generate` | POST | CustomVoice 模式生成 |
| `/api/v1/tts/qwen_tts/design` | POST | VoiceDesign 模式生成 |
| `/api/v1/tts/qwen_tts/clone` | POST | VoiceClone 模式生成 |
| `/api/v1/tts/stream` | POST | 统一流式端点 |

---

## 下一步

1. **查看完整文档**: `docs/QWEN3_TTS_DEPLOYMENT.md`
2. **运行测试**: `python backend/test_qwen_tts.py`
3. **集成到应用**: 参考 API 文档和示例代码

---

## 技术支持

- **GitHub Issues**: [提交问题](https://github.com/QwenLM/Qwen3-TTS/issues)
- **Qwen3-TTS 文档**: https://github.com/QwenLM/Qwen3-TTS
- **vLLM-Omni 文档**: https://docs.vllm.ai/projects/vllm-omni
