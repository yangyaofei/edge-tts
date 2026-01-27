"""
应用配置模块

本模块负责加载和管理应用配置，使用 Pydantic Settings 进行配置验证。
配置从多个来源加载（环境变量 > .env 文件 > 默认值）。
"""

from pydantic_settings import BaseSettings
from typing import Literal

import secrets
import os
from pathlib import Path


# ===================================================================
# 配置文件路径管理
# ===================================================================

def get_config_path() -> Path:
    """
    获取配置文件路径（支持多种部署场景）

    搜索顺序：
    1. config/config.env - Docker 部署或从项目根目录运行
    2. ../config/config.env - 从 backend/ 目录本地开发
    3. 根据 CWD 自动选择

    Returns:
        Path: 配置文件路径
    """
    # 1. Try explicit path (Docker or Root)
    p1 = Path("config/config.env")
    if p1.exists():
        return p1

    # 2. Try parent path (Local dev from backend/)
    p2 = Path("../config/config.env")
    if p2.exists():
        return p2

    # 3. Default for creation: Prefer ../config if we seem to be in backend/
    if Path.cwd().name == "backend":
        return Path("../config/config.env")

    return p1

config_path = get_config_path()


# ===================================================================
# 密钥管理
# ===================================================================

def get_or_create_secret():
    """
    获取或创建 SECRET_KEY

    SECRET_KEY 用于 JWT token 签名，必须保密。
    如果配置文件中不存在，会自动生成一个新的随机密钥。

    Returns:
        str | None: 新生成的密钥，如果已存在则返回 None
    """
    env_path = config_path

    # Ensure config directory exists
    try:
        env_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass # Might be permissions or read-only, we'll try to read/write anyway

    if env_path.exists():
        content = env_path.read_text().strip()
        if "SECRET_KEY=" in content:
            return None

    # If not exists or key missing, create/append
    secret = secrets.token_urlsafe(32)
    try:
        with open(env_path, "a") as f:
            f.write(f"\nSECRET_KEY={secret}\n")
        print(f"INFO: Generated new SECRET_KEY and saved to {env_path}")
    except Exception as e:
        print(f"WARN: Could not save SECRET_KEY to {env_path}: {e}")
        # We continue, assuming the app might run with ephemeral secret or fail later if strictly required
    return secret

# Ensure .env exists with a key before Settings loads
get_or_create_secret()

from typing import Optional


# ===================================================================
# 应用配置类
# ===================================================================

class Settings(BaseSettings):
    """
    应用配置类

    使用 Pydantic Settings 管理所有配置项。
    配置优先级：环境变量 > .env 文件 > 默认值
    """

    # -------------------------------------------------------------------
    # 基础配置
    # -------------------------------------------------------------------
    PROJECT_NAME: str = "TTS Bundles API"
    API_V1_STR: str = "/api/v1"

    # -------------------------------------------------------------------
    # 安全配置
    # -------------------------------------------------------------------
    # SECRET_KEY 用于 JWT token 签名，从 .env 文件加载
    SECRET_KEY: str

    # Access token 过期时间（分钟），None 表示永不过期
    ACCESS_TOKEN_EXPIRE_MINUTES: Optional[int] = None

    # -------------------------------------------------------------------
    # CORS 配置
    # -------------------------------------------------------------------
    # 允许的跨域来源，生产环境应限制具体域名
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    # ===================================================================
    # Qwen3-TTS 配置
    # ===================================================================
    # 是否启用 Qwen3-TTS 引擎
    QWEN_ENABLE: bool = True

    # 模型类型
    # - CustomVoice: 预设 9 种说话人，支持指令控制（推荐）
    # - Base: 声音克隆模型（未启用）
    # - VoiceDesign: 自然语言声音设计（未启用）
    QWEN_MODEL_TYPE: Literal["CustomVoice", "Base", "VoiceDesign"] = "CustomVoice"

    # 模型大小
    # - 0.6B: 轻量级模型，适合资源受限环境
    # - 1.7B: 高质量模型，推荐用于生产环境
    QWEN_MODEL_SIZE: Literal["0.6B", "1.7B"] = "1.7B"

    # 设备选择
    # - None: 自动检测（CUDA > MPS > CPU）
    # - "cuda:0": 强制使用第一块 NVIDIA GPU
    # - "mps": 强制使用 Apple Silicon GPU
    # - "cpu": 强制使用 CPU（速度慢）
    QWEN_DEVICE: Optional[str] = None

    # 最大生成 token 数
    # - 短文本（< 50 字）：2048
    # - 长文本（> 100 字）：4096
    QWEN_MAX_NEW_TOKENS: int = 2048

    # -------------------------------------------------------------------
    # Hugging Face 配置
    # -------------------------------------------------------------------
    # Hugging Face 访问令牌（用于下载私有模型）
    HF_TOKEN: Optional[str] = None

    # Hugging Face 镜像端点（用于加速模型下载）
    # 例如：https://hf-mirror.com
    HF_ENDPOINT: Optional[str] = None

    class Config:
        case_sensitive = True
        # env_file is now handled dynamically in instantiation


# 全局配置实例
settings = Settings(_env_file=str(config_path))
