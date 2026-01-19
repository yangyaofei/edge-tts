from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.endpoints import tts, text

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
    # Generate a token with no expiration (never expire)
    token = create_access_token(
        data={"sub": "admin", "admin": True}
    )
    print("\n" + "="*60)
    print(f"INFO:  Generated Admin Token (Never Expires):")
    print(f"{token}")
    print("="*60 + "\n")

@app.get("/")
def root():
    return {"message": "Welcome to TTS Bundles API"}
