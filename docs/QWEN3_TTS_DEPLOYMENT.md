# Qwen3-TTS 部署与集成指南

> **更新时间**: 2026-01-26
>
> 本文档详细说明如何部署 Qwen3-TTS 并集成到现有的 FastAPI TTS 服务中。

---

## 目录

1. [项目概述](#项目概述)
2. [Qwen3-TTS 简介](#qwen3-tts-简介)
3. [部署方案对比](#部署方案对比)
4. [方案一：使用 qwen-tts 包（推荐用于快速集成）](#方案一使用-qwen-tts-包推荐用于快速集成)
5. [方案二：使用 vLLM-Omni（推荐用于生产环境）](#方案二使用-vllm-omni推荐用于生产环境)
6. [集成到现有 FastAPI 服务](#集成到现有-fastapi-服务)
7. [验证测试](#验证测试)
8. [常见问题](#常见问题)

---

## 项目概述

现有项目是一个基于 FastAPI 的 TTS 服务，目前支持 Edge TTS 引擎。本文档将指导你如何添加 Qwen3-TTS 作为新的语音合成引擎。

### 当前项目结构

```
tts-bundles/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/
│   │   │   └── tts.py          # TTS API 端点
│   │   ├── services/
│   │   │   └── edge_engine.py  # Edge TTS 引擎实现
│   │   ├── schemas/
│   │   │   └── tts.py          # 数据模型
│   │   └── main.py             # FastAPI 应用入口
│   └── requirements.txt
├── extension/                  # Chrome 插件
└── docs/
```

---

## Qwen3-TTS 简介

Qwen3-TTS 是阿里云 Qwen 团队开发的开源文本转语音模型系列，具有以下特点：

### 核心特性

- **多语言支持**: 中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语（10种语言）
- **超低延迟**: 端到端合成延迟低至 97ms，支持流式生成
- **多种模式**:
  - **CustomVoice**: 预设 9 种高质量说话人音色
  - **VoiceDesign**: 通过自然语言描述设计声音
  - **Base**: 声音克隆（3秒音频即可克隆）
- **指令控制**: 支持通过自然语言指令控制语调、情感、韵律

### 可用模型

| 模型名称 | 特性 | 参数量 | 推荐场景 |
|---------|------|--------|---------|
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 9种预设音色 + 指令控制 | 1.7B | 通用TTS，需要多种音色选择 |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 自然语言声音设计 | 1.7B | 需要自定义声音特征 |
| Qwen3-TTS-12Hz-1.7B-Base | 声音克隆 | 1.7B | 需要克隆特定人声 |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 9种预设音色 | 0.6B | 资源受限环境 |
| Qwen3-TTS-12Hz-0.6B-Base | 声音克隆 | 0.6B | 资源受限环境的声音克隆 |

### 预设说话人（CustomVoice 模型）

| 说话人 | 声音描述 | 原生语言 |
|-------|---------|---------|
| Vivian | 明亮、略带棱角的年轻女性声音 | 中文 |
| Serena | 温柔、温和的年轻女性声音 | 中文 |
| Uncle_Fu | 成熟、低沉醇厚的男声 | 中文 |
| Dylan | 清晰自然的北京腔年轻男声 | 中文（北京方言） |
| Eric | 活泼、略带沙哑明亮的成都男声 | 中文（四川方言） |
| Ryan | 动感十足、节奏感强的男声 | 英语 |
| Aiden | 阳光开朗的美国男声 | 英语 |
| Ono_Anna | 顽皮轻盈的日本女性声音 | 日语 |
| Sohee | 温暖情感丰富的韩国女性声音 | 韩语 |

---

## 部署方案对比

### 方案对比表

| 特性 | qwen-tts 包 | vLLM-Omni |
|-----|------------|-----------|
| **部署难度** | ⭐ 简单 | ⭐⭐⭐ 中等 |
| **推理性能** | ⭐⭐⭐ 良好 | ⭐⭐⭐⭐⭐ 优秀 |
| **并发能力** | ⭐⭐ 有限 | ⭐⭐⭐⭐⭐ 强大 |
| **内存优化** | ⭐⭐⭐ 基础 | ⭐⭐⭐⭐⭐ 先进（PagedAttention） |
| **流式支持** | ✅ 支持 | ⚠️ 目前仅离线推理 |
| **API 集成** | ✅ 简单 | ⚠️ 需要额外服务 |
| **生产就绪** | ⭐⭐⭐ 适合小规模 | ⭐⭐⭐⭐⭐ 适合大规模 |
| **GPU 要求** | 8GB+ VRAM | 16GB+ VRAM 推荐 |

### 推荐选择

- **快速验证/开发环境**: 使用 **qwen-tts 包**（方案一）
- **生产环境/高并发**: 使用 **vLLM-Omni**（方案二）
- **资源受限**: 使用 **0.6B 模型 + qwen-tts 包**

---

## 方案一：使用 qwen-tts 包（推荐用于快速集成）

### 环境要求

- **操作系统**: Linux 或 macOS（Windows 需 WSL2）
- **Python**: 3.12
- **GPU**: NVIDIA GPU，8GB+ VRAM（compute capability 7.0+）
- **CUDA**: 11.8 或 12.x
- **RAM**: 16GB+ 推荐

### 安装步骤

#### 1. 创建虚拟环境

```bash
# 使用 conda
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# 或使用 venv
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/macOS
```

#### 2. 安装依赖

```bash
# 安装 qwen-tts 包
pip install -U qwen-tts

# （可选）安装 FlashAttention 2 以减少内存占用
pip install -U flash-attn --no-build-isolation

# 如果 RAM < 96GB，限制并行编译任务数
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

#### 3. 配置 Hugging Face 镜像（可选，中国大陆推荐）

```bash
# 使用 ModelScope 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

#### 4. 验证安装

```bash
python -c "from qwen_tts import Qwen3TTSModel; print('安装成功！')"
```

### 基本使用示例

#### CustomVoice 模式（推荐用于通用 TTS）

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 1. 加载模型
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",  # 如果安装了 FlashAttention
)

# 2. 生成语音
wavs, sr = model.generate_custom_voice(
    text="你好，这是一个测试。",
    language="Chinese",
    speaker="Vivian",  # 可选: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
    instruct="用温柔的声音说",  # 可选：风格指令
    max_new_tokens=2048,
)

# 3. 保存音频
sf.write("output.wav", wavs[0], sr)
```

#### VoiceDesign 模式

```python
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

wavs, sr = model.generate_voice_design(
    text="Hello! How are you today?",
    language="English",
    instruct="Speak in a cheerful and energetic tone",
    max_new_tokens=2048,
)
```

#### VoiceClone 模式

```python
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

wavs, sr = model.generate_voice_clone(
    text="这是克隆的声音在说话。",
    language="Chinese",
    ref_audio="reference.wav",  # 参考音频文件路径
    ref_text="参考音频的文字内容",
    max_new_tokens=2048,
)
```

---

## 方案二：使用 vLLM-Omni（推荐用于生产环境）

### 环境要求

- **操作系统**: Linux（暂不支持 Windows）
- **Python**: 3.12
- **GPU**: NVIDIA GPU，16GB+ VRAM 推荐（compute capability 7.0+）
- **CUDA**: 12.9（vLLM 0.14.0 基于 PyTorch 2.9.0）
- **RAM**: 32GB+ 推荐

### 安装步骤

#### 1. 安装 vLLM 和 vLLM-Omni

```bash
# 创建环境
conda create -n vllm-omni python=3.12 -y
conda activate vllm-omni

# 安装 uv（快速 Python 包管理器）
pip install uv

# 安装 vLLM
uv pip install vllm==0.14.0 --torch-backend=auto

# 克隆并安装 vLLM-Omni
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

#### 2. 使用 Docker 部署（推荐）

```bash
# 拉取官方镜像
docker pull vllm/vllm-omni:v0.14.0

# 启动 vLLM-Omni 服务器
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8091:8091 \
    --ipc=host \
    vllm/vllm-omni:v0.14.0 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --port 8091
```

### 使用示例

#### Python 客户端

```python
from vllm.omni import ChatCompletion

# 初始化客户端
client = ChatCompletion(
    base_url="http://localhost:8091/v1",
    api_key="empty",  # vLLM 默认不需要 API key
)

# 生成语音
response = client.completions.create(
    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    prompt="你好，这是使用 vLLM-Omni 生成的语音。",
    extra_body={
        "speaker": "Vivian",
        "language": "Chinese",
        "instruct": "用温柔的声音说",
    }
)

# 保存音频
with open("output_vllm.wav", "wb") as f:
    f.write(response.choices[0].audio.content)
```

#### 离线推理（直接使用 vLLM-Omni）

```bash
# 克隆 vllm-omni 仓库并进入示例目录
cd vllm-omni/examples/offline_inference/qwen3_tts

# CustomVoice 单样本
python end2end.py --query-type CustomVoice

# CustomVoice 批量样本
python end2end.py --query-type CustomVoice --use-batch-sample

# VoiceDesign 单样本
python end2end.py --query-type VoiceDesign

# Base 模型声音克隆
python end2end.py --query-type Base --mode-tag icl
```

### OpenAI API 兼容接口

vLLM-Omni 提供了 OpenAI API 兼容的接口：

```bash
curl http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "Hello, this is a test.",
    "voice": "Vivian",
    "language": "English"
  }' \
  --output speech.wav
```

---

## 集成到现有 FastAPI 服务

### 步骤 1：更新依赖

在 `backend/requirements.txt` 中添加：

```txt
# Qwen3-TTS 相关依赖
qwen-tts>=0.1.0
torch>=2.0.0
soundfile>=0.12.0
numpy>=1.24.0

# 如果使用 vLLM-Omni，添加：
vllm>=0.14.0
vllm-omni>=0.1.0
```

### 步骤 2：创建 Qwen TTS 引擎服务

创建文件 `backend/app/services/qwen_engine.py`：

```python
import torch
import logging
import io
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger(__name__)

class QwenTTSEngine:
    _model: Optional[Qwen3TTSModel] = None
    _model_type: Optional[str] = None

    @classmethod
    async def initialize(
        cls,
        model_type: str = "CustomVoice",
        model_size: str = "1.7B",
        device: str = "cuda:0"
    ):
        """初始化 Qwen3-TTS 模型"""
        if cls._model is not None:
            logger.warning("Model already initialized")
            return

        model_name = f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"
        logger.info(f"Loading Qwen3-TTS model: {model_name}")

        try:
            cls._model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=device,
                dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
            )
            cls._model_type = model_type
            logger.info("Qwen3-TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS model: {e}")
            raise

    @classmethod
    async def get_model(cls) -> Qwen3TTSModel:
        """获取已加载的模型"""
        if cls._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return cls._model

    @classmethod
    async def generate_custom_voice(
        cls,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: Optional[str] = None,
    ) -> tuple[bytes, int]:
        """
        使用 CustomVoice 模式生成语音

        返回: (audio_data, sample_rate)
        """
        model = await cls.get_model()

        try:
            import soundfile as sf
            import numpy as np

            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                max_new_tokens=2048,
            )

            # 将 numpy 数组转换为 WAV 格式的字节流
            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sr, format='WAV')
            audio_bytes = buffer.getvalue()
            buffer.close()

            return audio_bytes, sr

        except Exception as e:
            logger.error(f"Error in generate_custom_voice: {e}")
            raise

    @classmethod
    async def generate_voice_clone(
        cls,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: str = "Chinese",
        x_vector_only: bool = False,
    ) -> tuple[bytes, int]:
        """
        使用 Base 模型进行声音克隆

        返回: (audio_data, sample_rate)
        """
        model = await cls.get_model()

        try:
            import soundfile as sf

            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_path,
                ref_text=ref_text if not x_vector_only else None,
                x_vector_only_mode=x_vector_only,
                max_new_tokens=2048,
            )

            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sr, format='WAV')
            audio_bytes = buffer.getvalue()
            buffer.close()

            return audio_bytes, sr

        except Exception as e:
            logger.error(f"Error in generate_voice_clone: {e}")
            raise

    @classmethod
    def get_supported_speakers(cls) -> list[str]:
        """获取支持的说话人列表"""
        return [
            "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
            "Ryan", "Aiden", "Ono_Anna", "Sohee"
        ]

    @classmethod
    def get_supported_languages(cls) -> list[str]:
        """获取支持的语言列表"""
        return [
            "Auto", "Chinese", "English", "Japanese", "Korean",
            "French", "German", "Spanish", "Portuguese", "Russian"
        ]

@asynccontextmanager
async def get_qwen_model():
    """上下文管理器，用于确保模型在使用前已初始化"""
    try:
        yield await QwenTTSEngine.get_model()
    except RuntimeError as e:
        logger.error(f"Qwen model not initialized: {e}")
        raise
```

### 步骤 3：更新数据模型

在 `backend/app/schemas/tts.py` 中添加：

```python
from pydantic import BaseModel
from typing import List, Optional

# ... 现有的 schemas ...

# Qwen3-TTS 相关 Schemas
class QwenTTSRequest(BaseModel):
    text: str
    speaker: str = "Vivian"  # Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
    language: str = "Chinese"
    instruct: Optional[str] = None  # 可选的风格指令

class QwenVoiceCloneRequest(BaseModel):
    text: str
    ref_audio_url: str  # 参考音频的 URL 或 base64
    ref_text: str
    language: str = "Chinese"
    x_vector_only: bool = False

class QwenVoiceInfo(BaseModel):
    id: str
    name: str
    description: str
    native_language: str
```

### 步骤 4：更新 TTS 端点

在 `backend/app/api/v1/endpoints/tts.py` 中添加 Qwen 相关端点：

```python
from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File
from fastapi.responses import StreamingResponse, Response
from app.schemas.tts import EdgeTTSRequest, VoiceInfo, QwenTTSRequest, QwenVoiceCloneRequest
from app.services.edge_engine import EdgeTTSEngine
from app.services.qwen_engine import QwenTTSEngine, get_qwen_model
from app.core.security import verify_token

router = APIRouter()

# ... 现有的 Edge TTS 端点 ...

@router.get("/voices", response_model=list[VoiceInfo], dependencies=[Depends(verify_token)])
async def get_voices(engine: str = Query("edge", pattern="^(edge|qwen)$")):
    """获取支持的语音列表"""
    if engine == "edge":
        voices = await EdgeTTSEngine.get_voices()
        return [
            VoiceInfo(
                id=v["Name"],
                name=v["FriendlyName"],
                engine="edge",
                gender=v.get("Gender"),
                locale=v.get("Locale")
            ) for v in voices
        ]
    elif engine == "qwen":
        # 返回 Qwen 预设说话人
        speakers_data = {
            "Vivian": ("明亮、略带棱角的年轻女性声音", "中文"),
            "Serena": ("温柔、温和的年轻女性声音", "中文"),
            "Uncle_Fu": ("成熟、低沉醇厚的男声", "中文"),
            "Dylan": ("清晰自然的北京腔年轻男声", "中文（北京方言）"),
            "Eric": ("活泼、略带沙哑明亮的成都男声", "中文（四川方言）"),
            "Ryan": ("动感十足、节奏感强的男声", "英语"),
            "Aiden": ("阳光开朗的美国男声", "英语"),
            "Ono_Anna": ("顽皮轻盈的日本女性声音", "日语"),
            "Sohee": ("温暖情感丰富的韩国女性声音", "韩语"),
        }
        return [
            VoiceInfo(
                id=speaker,
                name=speaker,
                engine="qwen",
                description=desc,
                locale=lang
            )
            for speaker, (desc, lang) in speakers_data.items()
        ]
    return []

@router.post("/qwen/generate", dependencies=[Depends(verify_token)])
async def qwen_generate(request: QwenTTSRequest):
    """
    使用 Qwen3-TTS CustomVoice 模式生成语音

    参数:
    - text: 要转换的文本
    - speaker: 说话人 (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee)
    - language: 语言 (Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Auto)
    - instruct: 可选的风格指令，如 "用温柔的声音说"
    """
    try:
        audio_bytes, sr = await QwenTTSEngine.generate_custom_voice(
            text=request.text,
            speaker=request.speaker,
            language=request.language,
            instruct=request.instruct,
        )
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/qwen/clone", dependencies=[Depends(verify_token)])
async def qwen_clone(request: QwenVoiceCloneRequest):
    """
    使用 Qwen3-TTS Base 模型进行声音克隆

    参数:
    - text: 要转换的文本
    - ref_audio_url: 参考音频的 URL 或 base64 编码
    - ref_text: 参考音频的文字内容
    - language: 语言
    - x_vector_only: 是否仅使用 x-vector（不需要 ref_text，但质量会降低）
    """
    try:
        # TODO: 处理 ref_audio_url（支持 URL、base64 或文件上传）
        audio_bytes, sr = await QwenTTSEngine.generate_voice_clone(
            text=request.text,
            ref_audio_path=request.ref_audio_url,
            ref_text=request.ref_text,
            language=request.language,
            x_vector_only=request.x_vector_only,
        )
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 统一的流式端点（支持 engine 参数选择）
@router.post("/stream", dependencies=[Depends(verify_token)])
async def tts_stream(request: TTSRequest):
    """
    统一的 TTS 流式端点
    """
    try:
        if request.engine == "edge":
            audio_generator = EdgeTTSEngine.generate_stream(
                request.text,
                request.voice,
                request.rate,
                request.pitch
            )
            return StreamingResponse(audio_generator, media_type="audio/mpeg")
        elif request.engine == "qwen":
            # Qwen 暂不支持流式，返回完整音频
            # 解析 voice 字段获取 speaker 和 language
            audio_bytes, sr = await QwenTTSEngine.generate_custom_voice(
                text=request.text,
                speaker=request.voice,  # 使用 voice 字段作为 speaker
                language="Chinese",     # 默认中文，可扩展
            )
            return Response(content=audio_bytes, media_type="audio/wav")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported engine: {request.engine}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 步骤 5：在应用启动时初始化模型

在 `backend/app/main.py` 中添加启动事件：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.endpoints import tts, text
from app.services.qwen_engine import QwenTTSEngine
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(tts.router, prefix=f"{settings.API_V1_STR}/tts", tags=["tts"])
app.include_router(text.router, prefix=f"{settings.API_V1_STR}/text", tags=["text"])

from datetime import timedelta
from app.core.security import create_access_token

@app.on_event("startup")
async def startup_event():
    # 生成管理员 Token
    token = create_access_token(
        data={"sub": "admin", "admin": True}
    )
    print("\n" + "="*60)
    print(f"INFO:  Generated Admin Token (Never Expires):")
    print(f"{token}")
    print("="*60 + "\n")

    # 初始化 Qwen3-TTS 模型
    try:
        await QwenTTSEngine.initialize(
            model_type="CustomVoice",  # 或 "Base", "VoiceDesign"
            model_size="1.7B",         # 或 "0.6B"
            device="cuda:0"
        )
        logger.info("Qwen3-TTS initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Qwen3-TTS: {e}")
        logger.warning("Qwen3-TTS features will be unavailable")

@app.get("/")
def root():
    return {"message": "Welcome to TTS Bundles API"}
```

### 步骤 6：配置文件更新

在 `backend/app/core/config.py` 中添加 Qwen 相关配置：

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ... 现有配置 ...

    # Qwen3-TTS 配置
    QWEN_MODEL_TYPE: str = "CustomVoice"  # CustomVoice, Base, VoiceDesign
    QWEN_MODEL_SIZE: str = "1.7B"         # 0.6B, 1.7B
    QWEN_DEVICE: str = "cuda:0"
    QWEN_ENABLE: bool = True               # 是否启用 Qwen3-TTS

    class Config:
        env_file = ".env"

settings = Settings()
```

创建 `backend/.env` 文件：

```env
# Qwen3-TTS 配置
QWEN_MODEL_TYPE=CustomVoice
QWEN_MODEL_SIZE=1.7B
QWEN_DEVICE=cuda:0
QWEN_ENABLE=true

# Hugging Face 配置
HF_TOKEN=your_hugging_face_token_here
HF_ENDPOINT=https://hf-mirror.com
```

---

## 验证测试

### 测试 1：基本语音生成

```bash
# 启动服务
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 测试语音生成
curl -X POST "http://localhost:8000/api/v1/tts/qwen/generate" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是 Qwen3-TTS 的测试语音。",
    "speaker": "Vivian",
    "language": "Chinese",
    "instruct": "用温柔的声音说"
  }' \
  --output test.wav

# 播放生成的音频
ffplay test.wav  # 或使用其他音频播放器
```

### 测试 2：获取语音列表

```bash
curl -X GET "http://localhost:8000/api/v1/tts/voices?engine=qwen" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### 测试 3：多语言测试

```bash
# 英语
curl -X POST "http://localhost:8000/api/v1/tts/qwen/generate" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello! This is a test of Qwen3-TTS.",
    "speaker": "Ryan",
    "language": "English"
  }' \
  --output test_en.wav

# 日语
curl -X POST "http://localhost:8000/api/v1/tts/qwen/generate" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "こんにちは、これはテストです。",
    "speaker": "Ono_Anna",
    "language": "Japanese"
  }' \
  --output test_ja.wav
```

### 测试 4：不同说话人测试

```bash
# 测试所有说话人
for speaker in Vivian Serena Uncle_Fu Dylan Eric Ryan Aiden Ono_Anna Sohee; do
  echo "Testing speaker: $speaker"
  curl -X POST "http://localhost:8000/api/v1/tts/qwen/generate" \
    -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
      \"text\": \"你好，我是$speaker，很高兴认识你。\",
      \"speaker\": \"$speaker\",
      \"language\": \"Chinese\"
    }" \
    --output "test_${speaker}.wav"
done
```

### 测试 5：性能测试

```python
import time
import requests

token = "YOUR_ADMIN_TOKEN"
url = "http://localhost:8000/api/v1/tts/qwen/generate"

# 测试不同文本长度的生成时间
test_texts = [
    "短文本测试。",
    "这是一段中等长度的文本测试，用于评估 Qwen3-TTS 模型的性能表现和语音质量。",
    "这是一段较长的文本测试内容。在实际应用场景中，我们经常需要处理各种长度的文本输入。"
    "Qwen3-TTS 模型通过其先进的架构设计，能够高效地处理这些不同长度的文本，并生成高质量的语音输出。"
    "同时，模型还支持流式生成，可以在文本输入的同时就开始输出语音，大大降低了端到端的延迟。"
]

for i, text in enumerate(test_texts):
    start = time.time()
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={
            "text": text,
            "speaker": "Vivian",
            "language": "Chinese"
        }
    )
    end = time.time()

    if response.status_code == 200:
        duration = end - start
        print(f"Test {i+1}: {len(text)} chars, {duration:.2f}s, {len(response.content)/1024:.1f} KB")
    else:
        print(f"Test {i+1} failed: {response.status_code}")
```

---

## 常见问题

### Q1: GPU 内存不足怎么办？

**解决方案:**

1. **使用更小的模型**（0.6B）
   ```python
   await QwenTTSEngine.initialize(model_size="0.6B")
   ```

2. **量化模型**
   ```python
   model = Qwen3TTSModel.from_pretrained(
       model_name,
       device_map="cuda:0",
       dtype=torch.float16,  # 使用 fp16 而非 bf16
       load_in_8bit=True,    # 8-bit 量化
   )
   ```

3. **使用 CPU 推理**（速度慢但无内存限制）
   ```python
   model = Qwen3TTSModel.from_pretrained(
       model_name,
       device_map="cpu",
   )
   ```

### Q2: 模型下载速度慢或失败？

**解决方案:**

1. **使用 ModelScope 镜像**（中国大陆推荐）
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. **手动下载模型**
   ```bash
   # 使用 ModelScope
   pip install modelscope
   modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./models/qwen-tts

   # 然后从本地加载
   model = Qwen3TTSModel.from_pretrained("./models/qwen-tts")
   ```

3. **设置 Hugging Face 镜像环境变量**
   ```bash
   export HF_HUB_ENABLE_HF_TRANSFER=1
   ```

### Q3: 如何实现流式输出？

**说明:**

qwen-tts 包目前主要支持生成完整音频后再返回。虽然模型支持流式生成，但 Python API 的流式接口相对复杂。

**临时解决方案:**

对于长文本，可以先分块再生成：

```python
from app.services.chunker import TextChunker

async def generate_long_text_stream(text: str, speaker: str = "Vivian"):
    """将长文本分块后流式生成"""
    chunks = TextChunker.chunk_text(text, strategy="paragraph")

    for chunk in chunks:
        audio_bytes, sr = await QwenTTSEngine.generate_custom_voice(
            text=chunk.text,
            speaker=speaker,
            language="Chinese",
        )
        yield audio_bytes
```

### Q4: vLLM-Omni 与 qwen-tts 包如何选择？

**决策流程图:**

```
需要高并发 (>10 req/s)？
├─ 是 → 使用 vLLM-Omni
└─ 否
    │
    需要最低延迟？
    ├─ 是 → 使用 qwen-tts 包（单实例）
    └─ 否
        │
        GPU 资源充足 (16GB+)？
        ├─ 是 → 使用 vLLM-Omni（更好的扩展性）
        └─ 否 → 使用 qwen-tts 包 + 0.6B 模型
```

### Q5: 如何在生产环境部署？

**推荐架构:**

```
┌─────────────┐
│  Nginx/LB  │  (负载均衡)
└──────┬──────┘
       │
       ├──────────────────┐
       │                  │
┌──────▼──────┐    ┌─────▼─────┐
│  FastAPI    │    │  FastAPI  │  (多个实例)
│  Instance 1 │    │  Instance 2│
└──────┬──────┘    └─────┬─────┘
       │                  │
       └──────────┬───────┘
                  │
         ┌────────▼────────┐
         │  Qwen3-TTS      │  (共享模型或独立实例)
         │  (vLLM-Omni)    │
         └─────────────────┘
```

**Docker Compose 示例:**

```yaml
version: '3.8'

services:
  qwen-tts-server:
    image: vllm/vllm-omni:v0.14.0
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8091:8091"
    command: >
      --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
      --port 8091

  fastapi-backend:
    build: ./backend
    depends_on:
      - qwen-tts-server
    environment:
      - QWEN_SERVER_URL=http://qwen-tts-server:8091
    ports:
      - "8000:8000"
```

---

## 参考资料

- **Qwen3-TTS GitHub**: https://github.com/QwenLM/Qwen3-TTS
- **Qwen3-TTS 模型集合**: https://huggingface.co/collections/Qwen/qwen3-tts
- **vLLM-Omni 文档**: https://docs.vllm.ai/projects/vllm-omni
- **vLLM-Omni GitHub**: https://github.com/vllm-project/vllm-omni
- **Qwen3-TTS 技术报告**: https://arxiv.org/abs/2601.15621

---

## 更新日志

- **2026-01-26**: 初始版本，包含 qwen-tts 和 vLLM-Omni 两种部署方案
