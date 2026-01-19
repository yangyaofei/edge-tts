from pydantic_settings import BaseSettings

import secrets
import os
from pathlib import Path

# Helper to load or generate secret
def get_config_path() -> Path:
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

def get_or_create_secret():
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

class Settings(BaseSettings):
    PROJECT_NAME: str = "Edge TTS"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str # Will be loaded from .env
    ACCESS_TOKEN_EXPIRE_MINUTES: Optional[int] = None # Default to None (No expiration)

    # CORS Configuration
    BACKEND_CORS_ORIGINS: list[str] = ["*"] # Allow all for local dev, restrict in production

    class Config:
        case_sensitive = True
        # env_file is now handled dynamically in instantiation

settings = Settings(_env_file=str(config_path))
