from pydantic import BaseModel
from typing import List, Optional

# --- Text Processing Schemas ---
class TextChunkRequest(BaseModel):
    text: str
    strategy: str = "paragraph" # paragraph, sentence, punctuation

class Chunk(BaseModel):
    id: int
    text: str

class TextChunkResponse(BaseModel):
    chunks: List[Chunk]

# --- TTS Schemas ---
class EdgeTTSRequest(BaseModel):
    text: str
    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

class VoiceInfo(BaseModel):
    id: str
    name: str
    engine: str
    gender: Optional[str] = None
    locale: Optional[str] = None
