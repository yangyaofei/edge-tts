"""
安全认证模块

本模块提供 JWT (JSON Web Token) 认证功能，包括：
- Token 生成
- Token 验证
- 本地访问白名单

安全特性：
- JWT Token 认证
- 可配置的过期时间
- 本地访问豁免（127.0.0.1 和 ::1）
- Bearer Token 认证方式
"""

from datetime import datetime, timedelta
from typing import Optional
import jwt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

# Make auto_error=False to allow checking IP first
security = HTTPBearer(auto_error=False)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    创建 JWT 访问令牌

    Args:
        data: 要编码到 Token 中的数据（通常包含用户信息）
            例如：{"sub": "admin", "admin": True}
        expires_delta: 可选的过期时间增量
            - 如果提供，Token 将在指定时间后过期
            - 如果为 None，使用配置中的 ACCESS_TOKEN_EXPIRE_MINUTES
            - 如果配置也是 None，Token 永不过期

    Returns:
        str: 编码后的 JWT Token 字符串

    Example:
        >>> # 创建 1 小时后过期的 Token
        >>> token = create_access_token(
        ...     {"sub": "user123"},
        ...     expires_delta=timedelta(hours=1)
        ... )

        >>> # 创建永不过期的 Token
        >>> token = create_access_token({"sub": "admin", "admin": True})

    Note:
        Token 使用 HS256 算法签名，密钥来自 settings.SECRET_KEY。
    """
    to_encode = data.copy()

    # 设置过期时间
    if expires_delta:
        # 优先使用参数指定的过期时间
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire})
    elif settings.ACCESS_TOKEN_EXPIRE_MINUTES:
        # 其次使用配置文件中的过期时间
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
    # 如果都没有，则 Token 永不过期

    # 编码 JWT
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt


def verify_token(
    request: Request,
    token_auth: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    验证 JWT Token 或本地访问白名单

    这是 FastAPI 依赖函数，用于保护需要认证的端点。

    Args:
        request: FastAPI Request 对象，用于获取客户端 IP
        token_auth: 从 Authorization 头提取的 Bearer Token

    Returns:
        dict: Token 的 payload 数据
            - {"sub": "admin", "admin": True} - 认证通过
            - {"sub": "localhost", "admin": True} - 本地访问豁免

    Raises:
        HTTPException:
            - 403: 未提供 Token 且非本地访问
            - 401: Token 已过期
            - 401: Token 无效

    Example:
        >>> @router.get("/protected")
        >>> async def protected_route(user=Depends(verify_token)):
        ...     return {"message": f"Hello {user['sub']}"}

    Note:
        本地访问（127.0.0.1 或 ::1）自动豁免认证，方便开发和测试。
    """
    # 1. 本地访问白名单检查
    # ---------------------------------------------------------
    # 跳过 localhost 认证，方便本地开发
    client_host = request.client.host
    if client_host == "127.0.0.1" or client_host == "::1":
        # print(f"DEBUG: Bypassing auth for localhost: {client_host}")
        return {"sub": "localhost", "admin": True}

    # 2. Token 验证
    # ---------------------------------------------------------
    # 检查是否提供了 Token
    if not token_auth:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = token_auth.credentials

    # 解码并验证 Token
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        # Token 已过期
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        # Token 无效（签名错误、格式错误等）
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
