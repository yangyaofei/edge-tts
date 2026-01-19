from pydantic_settings import BaseSettings

import secrets
import os
from pathlib import Path

# Helper to load or generate secret
def get_or_create_secret():
    env_path = Path(".env")
    if env_path.exists():
        # Let pydantic load it
        return None
    
    # If not exists, create it
    secret = secrets.token_urlsafe(32)
    with open(env_path, "w") as f:
        f.write(f"SECRET_KEY={secret}\n")
    print(f"INFO: Generated new SECRET_KEY and saved to .env")
    return secret

# Ensure .env exists with a key before Settings loads
get_or_create_secret()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Edge TTS"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str # Will be loaded from .env
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30 # 30 Days

    # CORS Configuration
    BACKEND_CORS_ORIGINS: list[str] = ["*"] # Allow all for local dev, restrict in production

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
