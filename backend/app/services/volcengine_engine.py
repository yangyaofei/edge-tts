import asyncio
import io
import json
import logging
import struct
import uuid
from enum import IntEnum
from typing import AsyncGenerator

import websockets

logger = logging.getLogger(__name__)

ENDPOINT = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
RESOURCE_ID = "seed-tts-2.0"
SAMPLE_RATE = 24000

FORMAT_MAP = {
    "mp3": "mp3",
    "opus": "ogg_opus",
    "aac": "mp3",
    "flac": "mp3",
    "wav": "pcm",
    "pcm": "pcm",
}

VOICE_MAP = {
    "alloy": "zh_female_vv_uranus_bigtts",
    "echo": "zh_male_m191_uranus_bigtts",
    "shimmer": "zh_female_cancan_uranus_bigtts",
    "fable": "zh_female_vv_uranus_bigtts",
    "nova": "zh_female_cancan_uranus_bigtts",
    "onyx": "zh_male_m191_uranus_bigtts",
    "coral": "zh_female_cancan_uranus_bigtts",
    "sage": "zh_male_m191_uranus_bigtts",
    "ash": "zh_male_m191_uranus_bigtts",
    "ballad": "zh_female_vv_uranus_bigtts",
    "verse": "zh_male_m191_uranus_bigtts",
}

OPENAI_VOICES = [
    {"id": "alloy", "name": "Alloy (Vivi)", "engine": "volcengine", "locale": "zh-CN"},
    {"id": "echo", "name": "Echo (云舟)", "engine": "volcengine", "locale": "zh-CN"},
    {"id": "shimmer", "name": "Shimmer (灿灿)", "engine": "volcengine", "locale": "zh-CN"},
    {"id": "fable", "name": "Fable (Vivi)", "engine": "volcengine", "locale": "zh-CN"},
    {"id": "nova", "name": "Nova (灿灿)", "engine": "volcengine", "locale": "zh-CN"},
    {"id": "onyx", "name": "Onyx (云舟)", "engine": "volcengine", "locale": "zh-CN"},
    {"id": "coral", "name": "Coral (灿灿)", "engine": "volcengine", "locale": "zh-CN"},
    {"id": "sage", "name": "Sage (云舟)", "engine": "volcengine", "locale": "zh-CN"},
]


class MsgType(IntEnum):
    FullClientRequest = 0b1
    FullServerResponse = 0b1001
    AudioOnlyServer = 0b1011
    Error = 0b1111


class EventType(IntEnum):
    StartConnection = 1
    FinishConnection = 2
    ConnectionStarted = 50
    ConnectionFailed = 51
    ConnectionFinished = 52
    StartSession = 100
    FinishSession = 102
    SessionStarted = 150
    SessionFinished = 152
    SessionFailed = 153
    TaskRequest = 200


def _build_frame(event, session_id="", payload=b"{}"):
    buf = io.BytesIO()
    buf.write(bytes([(1 << 4) | 1, (MsgType.FullClientRequest << 4) | 0b100, (1 << 4) | 0, 0]))
    buf.write(struct.pack(">i", event))
    if event not in (EventType.StartConnection, EventType.FinishConnection):
        sid = session_id.encode("utf-8")
        buf.write(struct.pack(">I", len(sid)))
        if sid:
            buf.write(sid)
    buf.write(struct.pack(">I", len(payload)))
    buf.write(payload)
    return buf.getvalue()


def _parse(data):
    mt = data[1] >> 4
    flag = data[1] & 0x0F
    pos = 4
    event = struct.unpack(">i", data[pos:pos + 4])[0] if flag & 0b100 else 0
    pos += 4
    if flag & 0b100 and event not in (1, 2, 50, 51, 52):
        slen = struct.unpack(">I", data[pos:pos + 4])[0]
        pos += 4
        if slen:
            pos += slen
    if flag & 0b100 and event in (50, 51, 52):
        clen = struct.unpack(">I", data[pos:pos + 4])[0]
        pos += 4
        if clen:
            pos += clen
    payload = b""
    if pos + 4 <= len(data):
        plen = struct.unpack(">I", data[pos:pos + 4])[0]
        pos += 4
        if plen:
            payload = data[pos:pos + plen]
    return mt, event, payload


class VolcengineTTSEngine:
    def __init__(self, api_key: str = "", app_id: str = "", access_token: str = ""):
        self.api_key = api_key
        self.app_id = app_id
        self.access_token = access_token

    def _build_headers(self) -> dict:
        if self.api_key:
            return {
                "X-Api-Key": self.api_key,
                "X-Api-Resource-Id": RESOURCE_ID,
                "X-Api-Request-Id": str(uuid.uuid4()),
            }
        if self.app_id and self.access_token:
            return {
                "X-Api-App-Key": self.app_id,
                "X-Api-Access-Key": self.access_token,
                "X-Api-Resource-Id": RESOURCE_ID,
                "X-Api-Connect-Id": str(uuid.uuid4()),
            }
        raise RuntimeError("volcengine: no auth credentials configured")

    @staticmethod
    def resolve_voice(voice: str) -> str:
        return VOICE_MAP.get(voice, voice)

    @staticmethod
    def get_openai_voices() -> list:
        return OPENAI_VOICES

    async def generate_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        audio_format: str = "mp3",
    ) -> AsyncGenerator[bytes, None]:
        headers = self._build_headers()
        ws = await websockets.connect(ENDPOINT, additional_headers=headers, max_size=10 * 1024 * 1024)

        volc_format = FORMAT_MAP.get(audio_format, "mp3")
        speech_rate = max(-50, min(100, int((speed - 1.0) * 100)))

        try:
            await ws.send(_build_frame(EventType.StartConnection, payload=b"{}"))
            while True:
                d = await ws.recv()
                mt, ev, pl = _parse(d)
                if mt == MsgType.FullServerResponse and ev == EventType.ConnectionStarted:
                    break
                if mt == MsgType.Error or (mt == MsgType.FullServerResponse and ev == EventType.ConnectionFailed):
                    raise RuntimeError(f"volcengine connection failed: {pl.decode('utf-8', 'ignore')}")

            session_id = str(uuid.uuid4())
            req_params = {
                "speaker": voice,
                "audio_params": {"format": volc_format, "sample_rate": SAMPLE_RATE, "speech_rate": speech_rate},
                "additions": json.dumps({"disable_markdown_filter": True, "latex_parser": "v2"}),
            }
            session_req = {
                "user": {"uid": str(uuid.uuid4())},
                "namespace": "BidirectionalTTS",
                "event": EventType.StartSession,
                "req_params": req_params,
            }
            await ws.send(_build_frame(EventType.StartSession, session_id, json.dumps(session_req).encode()))
            while True:
                d = await ws.recv()
                mt, ev, pl = _parse(d)
                if mt == MsgType.FullServerResponse and ev == EventType.SessionStarted:
                    break
                if mt == MsgType.Error or (mt == MsgType.FullServerResponse and ev == EventType.SessionFailed):
                    raise RuntimeError(f"volcengine session failed: {pl.decode('utf-8', 'ignore')}")

            max_chunk = 800
            paragraphs = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                while len(line) > max_chunk:
                    paragraphs.append(line[:max_chunk])
                    line = line[max_chunk:]
                paragraphs.append(line)

            async def send_all():
                for para in paragraphs:
                    task_req = {
                        "user": {"uid": str(uuid.uuid4())},
                        "namespace": "BidirectionalTTS",
                        "event": EventType.TaskRequest,
                        "req_params": {**req_params, "text": para},
                    }
                    await ws.send(_build_frame(EventType.TaskRequest, session_id, json.dumps(task_req).encode()))
                    await asyncio.sleep(0.01)
                await ws.send(_build_frame(EventType.FinishSession, session_id, b"{}"))

            send_task = asyncio.create_task(send_all())

            while True:
                d = await ws.recv()
                mt, ev, pl = _parse(d)
                if mt == MsgType.AudioOnlyServer:
                    if pl:
                        yield pl
                elif mt == MsgType.FullServerResponse and ev == EventType.SessionFinished:
                    break
                elif mt == MsgType.Error:
                    raise RuntimeError(f"volcengine tts error: {pl.decode('utf-8', 'ignore')}")

            await send_task
        finally:
            await ws.send(_build_frame(EventType.FinishConnection, payload=b"{}"))
            try:
                await asyncio.wait_for(ws.recv(), timeout=3)
            except Exception:
                pass
            await ws.close()
