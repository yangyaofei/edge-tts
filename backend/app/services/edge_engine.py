import edge_tts
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class EdgeTTSEngine:
    @staticmethod
    async def get_voices() -> list:
        """
        Returns a list of available voices.
        """
        voices = await edge_tts.list_voices()
        return voices

    @staticmethod
    async def generate_stream(text: str, voice: str, rate: str, pitch: str) -> AsyncGenerator[bytes, None]:
        """
        Generates an audio stream (mp3) from text.
        """
        logger.debug(f"Starting generation for text: {text[:20]}...")
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        
        count = 0
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    count += 1
                    # logger.debug(f"Yielding audio chunk {count}, size={len(chunk['data'])}")
                    yield chunk["data"]
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            raise e
        
        logger.debug(f"Generation finished. Total audio chunks: {count}")
