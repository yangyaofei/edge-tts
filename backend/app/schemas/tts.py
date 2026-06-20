from pydantic import BaseModel
from typing import List, Optional, Union


class TextChunkRequest(BaseModel):
    text: str
    strategy: str = "paragraph"

class Chunk(BaseModel):
    id: int
    text: str

class TextChunkResponse(BaseModel):
    chunks: List[Chunk]


class TTSRequest(BaseModel):
    text: str
    engine: str = "edge"
    voice: str = "zh-CN-XiaoxiaoNeural"
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    preprocess: bool = True
    language: str = "zh"
    temperature: Optional[float] = None
    instruct: Optional[str] = None


class VoiceInfo(BaseModel):
    id: str
    name: str
    engine: str
    gender: Optional[str] = None
    locale: Optional[str] = None


class OpenAITTSRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: Union[str, dict] = "alloy"
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    stream_format: Optional[str] = "audio"
    instructions: Optional[str] = None

    class Config:
        populate_by_name = True


OPENAI_AUDIO_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}
