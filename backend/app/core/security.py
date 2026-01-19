from datetime import datetime, timedelta
from typing import Optional
import jwt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

# Make auto_error=False to allow checking IP first
security = HTTPBearer(auto_error=False)

def verify_token(request: Request, token_auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    # 1. Bypass for Localhost
    client_host = request.client.host
    if client_host == "127.0.0.1" or client_host == "::1":
        # print(f"DEBUG: Bypassing auth for localhost: {client_host}")
        return {"sub": "localhost", "admin": True}

    # 2. Check Token
    if not token_auth:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = token_auth.credentials
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
