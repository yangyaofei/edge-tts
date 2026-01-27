# Qwen3-TTS 集成完成总结

> **完成时间**: 2026-01-26
> **状态**: ✅ 完成并测试就绪

---

## 已完成的工作

### 1. 核心文件创建/修改

#### 新增文件

| 文件 | 说明 |
|------|------|
| `backend/app/services/qwen_engine.py` | Qwen3-TTS 引擎实现（支持 CUDA/MPS 自动检测） |
| `backend/test_qwen_tts.py` | 完整的测试脚本（8个测试场景） |
| `backend/.env.example` | 配置示例文件（含详细注释） |
| `docs/QWEN3_TTS_DEPLOYMENT.md` | 部署指南（完整技术文档） |
| `docs/QWEN_TTS_USAGE.md` | 使用指南（快速上手文档） |

#### 修改文件

| 文件 | 主要变更 |
|------|---------|
| `backend/app/schemas/tts.py` | 添加 Qwen TTS 请求模型（CustomVoice, VoiceDesign, VoiceClone, Unified） |
| `backend/app/api/v1/endpoints/tts.py` | 添加 Qwen TTS 端点（5个新端点 + 更新） |
| `backend/app/core/config.py` | 添加 Qwen 配置项（模型类型、大小、设备等） |
| `backend/app/main.py` | 添加启动时自动初始化 Qwen 模型 |
| `backend/requirements.txt` | 添加 Qwen 相关依赖 |

### 2. 功能特性

#### ✅ 设备支持

- **CUDA (NVIDIA GPU)**: 自动检测并使用，优先级最高
- **MPS (Apple Silicon)**: 自动检测 M1/M2/M3/M4，使用 float16
- **CPU**: 降级选项，显存不足时可用

#### ✅ 模型支持

- **CustomVoice**: 9 种预设说话人（默认模式）
- **VoiceDesign**: 自然语言声音设计
- **VoiceClone**: 3秒音频克隆声音

#### ✅ 模型大小

- **1.7B**: 高质量模型（默认）
- **0.6B**: 轻量级模型（资源受限时）

#### ✅ 多语言支持

中文、英语、日语、韩语、法语、德语、西班牙语、葡萄牙语、俄语（10种）

---

## API 端点

### 新增端点

```
GET  /api/v1/tts/voices?engine=qwen_tts           # 获取 Qwen 语音列表
GET  /api/v1/tts/health/qwen                      # Qwen 健康检查
POST /api/v1/tts/qwen_tts/generate                # CustomVoice 模式
POST /api/v1/tts/qwen_tts/design                  # VoiceDesign 模式
POST /api/v1/tts/qwen_tts/clone                   # VoiceClone 模式
```

### 更新端点

```
POST /api/v1/tts/stream                           # 支持 engine="qwen_tts"
GET  /health                                      # 返回 Qwen 状态
GET  /                                            # 返回引擎列表
```

---

## 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|-----|-------|------|
| `QWEN_ENABLE` | `true` | 是否启用 Qwen3-TTS |
| `QWEN_MODEL_TYPE` | `CustomVoice` | 模型类型 |
| `QWEN_MODEL_SIZE` | `1.7B` | 模型大小 |
| `QWEN_DEVICE` | `None` | 设备（None=自动检测） |
| `QWEN_MAX_NEW_TOKENS` | `2048` | 最大生成 token 数 |
| `HF_TOKEN` | `None` | Hugging Face token |
| `HF_ENDPOINT` | `None` | HF 镜像 URL |

### 设备自动检测逻辑

```python
if CUDA available:
    device = "cuda:0"
    dtype = bfloat16  # 更好性能
elif MPS available:
    device = "mps"
    dtype = float16   # 更稳定
else:
    device = "cpu"
    dtype = float32
```

---

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置

```bash
mkdir -p config
cp .env.example config/config.env
# 编辑 config/config.env（可选）
```

### 3. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 测试

```bash
# 设置 token（可选）
export TTS_TOKEN="your_admin_token"

# 运行测试
python test_qwen_tts.py
```

### 5. API 调用示例

```bash
curl -X POST "http://localhost:8000/api/v1/tts/qwen_tts/generate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是测试语音。",
    "speaker": "Vivian",
    "language": "Chinese"
  }' \
  --output test.wav
```

---

## 代码示例

### Python 客户端

```python
import requests

# 配置
BASE_URL = "http://localhost:8000"
TOKEN = "your_token"  # 可选，localhost 可能不需要

# 请求头
headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

# 生成语音
response = requests.post(
    f"{BASE_URL}/api/v1/tts/qwen_tts/generate",
    headers=headers,
    json={
        "text": "你好，这是 Qwen3-TTS 生成的语音。",
        "speaker": "Vivian",
        "language": "Chinese",
        "instruct": "用温柔的声音说"  # 可选
    }
)

# 保存音频
if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("✓ 音频已保存")
```

### 使用统一端点

```python
response = requests.post(
    f"{BASE_URL}/api/v1/tts/stream",
    headers=headers,
    json={
        "text": "测试文本",
        "engine": "qwen_tts",  # 选择引擎
        "speaker": "Vivian",
        "language": "Chinese"
    }
)
```

---

## 测试脚本功能

`test_qwen_tts.py` 包含 8 个测试场景：

1. **健康检查**: 验证服务状态和模型信息
2. **语音列表**: 获取所有可用说话人
3. **基本生成**: 测试默认配置生成
4. **多语言测试**: 测试中英日韩四种语言
5. **说话人测试**: 测试 5 个不同说话人
6. **风格指令**: 测试不同风格指令效果
7. **统一端点**: 测试流式端点
8. **长文本测试**: 测试长文本处理能力

运行：
```bash
python backend/test_qwen_tts.py
```

---

## 项目结构

```
tts-bundles/
├── backend/
│   ├── app/
│   │   ├── services/
│   │   │   ├── edge_engine.py      # Edge TTS 引擎
│   │   │   └── qwen_engine.py      # ✨ Qwen3-TTS 引擎（新增）
│   │   ├── api/v1/endpoints/
│   │   │   └── tts.py              # ✨ TTS API 端点（更新）
│   │   ├── schemas/
│   │   │   └── tts.py              # ✨ 数据模型（更新）
│   │   ├── core/
│   │   │   └── config.py           # ✨ 配置（更新）
│   │   └── main.py                 # ✨ 主应用（更新）
│   ├── requirements.txt            # ✨ 依赖（更新）
│   ├── test_qwen_tts.py            # ✨ 测试脚本（新增）
│   └── .env.example                # ✨ 配置示例（新增）
├── docs/
│   ├── QWEN3_TTS_DEPLOYMENT.md     # ✨ 部署指南（新增）
│   ├── QWEN_TTS_USAGE.md           # ✨ 使用指南（新增）
│   └── QWEN_INTEGRATION_SUMMARY.md # ✨ 本文档（新增）
└── config/
    └── config.env                  # 运行时生成
```

---

## 关键实现细节

### 1. 设备自动检测

```python
@classmethod
def get_available_device(cls) -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

### 2. 数据类型优化

```python
@classmethod
def get_optimal_dtype(cls, device: str) -> torch.dtype:
    if device == "mps":
        return torch.float16   # MPS 更稳定
    elif device.startswith("cuda"):
        return torch.bfloat16  # CUDA 更好性能
    else:
        return torch.float32   # CPU
```

### 3. 优雅的启动失败处理

```python
@app.on_event("startup")
async def startup_event():
    if settings.QWEN_ENABLE:
        try:
            await QwenTTSEngine.initialize(...)
            print("✓ Qwen3-TTS initialized successfully!")
        except Exception as e:
            logger.warning(f"Failed to initialize: {e}")
            print("⚠ Qwen3-TTS initialization failed!")
            # 服务继续运行，只是 Qwen 功能不可用
```

### 4. 类型提示

```python
from typing import Literal

ModelType = Literal["CustomVoice", "Base", "VoiceDesign"]
ModelSize = Literal["0.6B", "1.7B"]
Language = Literal["Auto", "Chinese", "English", ...]
```

---

## 性能参考

### 硬件要求

| 配置 | 显存/内存 | 推荐模型 | 预期速度 |
|-----|---------|---------|---------|
| NVIDIA GPU (8GB VRAM) | 8GB | 1.7B | ~5-10x 实时 |
| NVIDIA GPU (16GB VRAM) | 16GB | 1.7B | ~10-20x 实时 |
| Apple M1/M2 (16GB RAM) | 16GB | 1.7B | ~3-5x 实时 |
| Apple M3 (32GB RAM) | 32GB | 1.7B | ~5-10x 实时 |
| CPU (现代多核) | 32GB RAM | 0.6B | ~0.3-0.5x 实时 |

### 模型大小

| 模型 | 下载大小 | 内存占用 |
|-----|---------|---------|
| 1.7B | ~3.5GB | ~4-5GB |
| 0.6B | ~1.5GB | ~2-3GB |

---

## 已知限制

1. **MPS 后端**:
   - 部分算子可能不支持，会回退到 CPU
   - `bfloat16` 支持不完善，使用 `float16`

2. **模型加载**:
   - 首次启动需要下载模型（~3.5GB）
   - 加载时间取决于网络和磁盘速度

3. **并发限制**:
   - 当前实现为单模型实例
   - 高并发场景考虑使用 vLLM-Omni

---

## 未来改进

1. **流式生成**: 实现真正的流式音频输出
2. **批处理**: 支持批量文本处理
3. **vLLM-Omni 集成**: 支持更高并发
4. **MLX 后端**: Apple Silicon 原生优化
5. **缓存机制**: 重复文本的音频缓存

---

## 故障排查

### 问题：模型加载失败

```
Error: qwen-tts package not installed
```

**解决方案**:
```bash
pip install qwen-tts
```

### 问题：CUDA 不可用

```
Warning: No GPU detected, using CPU
```

**检查**:
```python
import torch
print(torch.cuda.is_available())  # 应该是 True
```

**解决方案**:
- 安装 CUDA 版 PyTorch
- 检查 NVIDIA 驱动

### 问题：MPS 错误

```
RuntimeError: MPS does not support ...
```

**解决方案**:
```env
QWEN_DEVICE=cpu  # 临时使用 CPU
```

---

## 相关资源

- **Qwen3-TTS GitHub**: https://github.com/QwenLM/Qwen3-TTS
- **vLLM-Omni 文档**: https://docs.vllm.ai/projects/vllm-omni
- **Hugging Face 模型**: https://huggingface.co/Qwen
- **PyTorch MPS 文档**: https://pytorch.org/docs/stable/notes/mps.html

---

## 总结

✅ **完整实现**: 从配置到 API 的完整集成
✅ **设备支持**: CUDA 和 MPS 自动检测
✅ **生产就绪**: 错误处理、日志、健康检查
✅ **文档齐全**: 部署、使用、测试文档
✅ **测试覆盖**: 8 个测试场景
✅ **向后兼容**: 不影响现有 Edge TTS 功能

**现在可以开始使用！** 🚀
