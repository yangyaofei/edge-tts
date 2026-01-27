"""
FastAPI 应用主入口模块

本模块创建并配置 FastAPI 应用实例，包括：
- CORS 中间件配置
- API 路由注册
- 启动事件处理（生成 Token、初始化 Qwen 模型）
- 健康检查端点
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.endpoints import tts, text
from app.services.qwen_engine import QwenTTSEngine
import logging

logger = logging.getLogger(__name__)


# ===================================================================
# FastAPI 应用实例
# ===================================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="支持 Edge TTS 和 Qwen3-TTS 的统一 API",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)


# ===================================================================
# CORS 中间件配置
# ===================================================================

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ===================================================================
# API 路由注册
# ===================================================================

app.include_router(tts.router, prefix=f"{settings.API_V1_STR}/tts", tags=["tts"])
app.include_router(text.router, prefix=f"{settings.API_V1_STR}/text", tags=["text"])

from datetime import timedelta
from app.core.security import create_access_token


# ===================================================================
# 启动事件处理
# ===================================================================

@app.on_event("startup")
async def startup_event():
    """
    应用启动时执行

    执行以下任务：
    1. 生成管理员访问 Token（永不过期）
    2. 初始化 Qwen3-TTS 模型（如果启用）
    """
    # -------------------------------------------------------------------
    # 1. 生成管理员 Token
    # -------------------------------------------------------------------
    # Generate a token with no expiration (never expire)
    token = create_access_token(
        data={"sub": "admin", "admin": True}
    )
    print("\n" + "="*60)
    print(f"INFO:  Generated Admin Token (Never Expires):")
    print(f"{token}")
    print("="*60 + "\n")

    # -------------------------------------------------------------------
    # 2. 初始化 Qwen3-TTS 模型
    # -------------------------------------------------------------------
    # Initialize Qwen3-TTS model if enabled
    if settings.QWEN_ENABLE:
        try:
            logger.info("Initializing Qwen3-TTS model...")
            logger.info(f"  Model type: {settings.QWEN_MODEL_TYPE}")
            logger.info(f"  Model size: {settings.QWEN_MODEL_SIZE}")
            logger.info(f"  Device: {settings.QWEN_DEVICE or 'auto-detect'}")

            await QwenTTSEngine.initialize(
                model_type=settings.QWEN_MODEL_TYPE,
                model_size=settings.QWEN_MODEL_SIZE,
                device=settings.QWEN_DEVICE,
            )

            print("\n" + "="*60)
            print("✓ Qwen3-TTS initialized successfully!")
            print(f"  Model: Qwen3-TTS-12Hz-{settings.QWEN_MODEL_SIZE}-{settings.QWEN_MODEL_TYPE}")
            print(f"  Device: {QwenTTSEngine.get_model_info()['device']}")
            print("="*60 + "\n")

        except Exception as e:
            logger.warning(f"Failed to initialize Qwen3-TTS: {e}")
            logger.warning("Qwen3-TTS features will be unavailable")
            logger.warning("To disable Qwen3-TTS, set QWEN_ENABLE=false in config")
            print("\n" + "="*60)
            print("⚠ Qwen3-TTS initialization failed!")
            print(f"  Error: {e}")
            print("  Qwen3-TTS features will be unavailable")
            print("  To disable this warning, set QWEN_ENABLE=false in config")
            print("="*60 + "\n")
    else:
        logger.info("Qwen3-TTS is disabled (QWEN_ENABLE=false)")


# ===================================================================
# 根路径端点
# ===================================================================

@app.get("/")
def root():
    """
    根路径端点

    返回 API 基本信息和支持的引擎列表。
    """
    return {
        "message": "Welcome to TTS Bundles API",
        "engines": ["edge", "qwen_tts"] if settings.QWEN_ENABLE else ["edge"],
        "qwen_enabled": settings.QWEN_ENABLE,
    }


# ===================================================================
# 健康检查端点
# ===================================================================

@app.get("/health")
async def health_check():
    """
    健康检查端点

    返回服务状态和各引擎的可用性。
    """
    return {
        "status": "healthy",
        "engines": {
            "edge": "enabled",
            "qwen_tts": "enabled" if settings.QWEN_ENABLE else "disabled",
        },
        "qwen_model": QwenTTSEngine.get_model_info() if settings.QWEN_ENABLE else None,
    }
