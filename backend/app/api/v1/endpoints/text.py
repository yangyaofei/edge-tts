from fastapi import APIRouter, Depends
from app.schemas.tts import TextChunkRequest, TextChunkResponse, Chunk
from app.services.chunker import TextChunker
from app.core.security import verify_token

router = APIRouter()

@router.post("/chunk", response_model=TextChunkResponse, dependencies=[Depends(verify_token)])
async def chunk_text(request: TextChunkRequest):
    chunks_text = TextChunker.chunk_text(request.text, request.strategy)
    chunks = [Chunk(id=i, text=t) for i, t in enumerate(chunks_text)]
    return TextChunkResponse(chunks=chunks)
