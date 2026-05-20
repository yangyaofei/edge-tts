#!/usr/bin/env python3
"""
火山引擎 TTS 验证脚本 - WebSocket 双向流式 API
将 Markdown 文档转为语音

用法:
  uv run tts_demo.py <markdown_file> [--output OUTPUT]
  uv run tts_demo.py --text "要合成的文本"
  uv run tts_demo.py doc.md --voice zh_female_vv_uranus_bigtts --speech-rate 10
"""
import argparse
import asyncio
import copy
import io
import json
import logging
import struct
import sys
import uuid
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable

import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ENDPOINT = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
RESOURCE_ID = "seed-tts-2.0"
DEFAULT_VOICE = "zh_female_vv_uranus_bigtts"
AUDIO_FORMAT = "mp3"
SAMPLE_RATE = 24000
MAX_CHARS_PER_SESSION = 800


class MsgType(IntEnum):
    FullClientRequest = 0b1
    AudioOnlyClient = 0b10
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
    CancelSession = 101
    FinishSession = 102
    SessionStarted = 150
    SessionCanceled = 151
    SessionFinished = 152
    SessionFailed = 153
    TaskRequest = 200
    TTSSentenceStart = 350
    TTSSentenceEnd = 351
    TTSResponse = 352


def _build_frame(event: EventType, session_id: str = "", payload: bytes = b"{}") -> bytes:
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


def _parse_response(data: bytes):
    msg_type = data[1] >> 4
    flag = data[1] & 0x0F
    pos = 4
    event = struct.unpack(">i", data[pos:pos + 4])[0] if flag & 0b100 else 0
    pos += 4
    session_id = ""
    if flag & 0b100 and event not in (
        EventType.StartConnection, EventType.FinishConnection,
        EventType.ConnectionStarted, EventType.ConnectionFailed,
        EventType.ConnectionFinished,
    ):
        sid_len = struct.unpack(">I", data[pos:pos + 4])[0]
        pos += 4
        if sid_len:
            session_id = data[pos:pos + sid_len].decode("utf-8")
            pos += sid_len
    if flag & 0b100 and event in (
        EventType.ConnectionStarted, EventType.ConnectionFailed,
        EventType.ConnectionFinished,
    ):
        pass
    payload = b""
    if pos + 4 <= len(data):
        plen = struct.unpack(">I", data[pos:pos + 4])[0]
        pos += 4
        if plen:
            payload = data[pos:pos + plen]
    return msg_type, event, session_id, payload


def load_env() -> dict:
    env = {}
    for env_path in [Path(__file__).parent / ".env", Path(__file__).parent / "config" / "config.env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env


def read_markdown(path: str) -> str:
    md = Path(path).read_text(encoding="utf-8")
    lines = md.splitlines()
    clean = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("```"):
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        if line.startswith("|"):
            continue
        if line.startswith("---"):
            continue
        if line.startswith("- ") or line.startswith("* "):
            line = line[2:].strip()
        if line.startswith("> "):
            line = line[2:].strip()
        clean.append(line)
    return "\n".join(clean)


def split_text(text: str, max_chars: int = MAX_CHARS_PER_SESSION) -> list[str]:
    paragraphs = text.split("\n")
    chunks = []
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) + 1 > max_chars:
            if current:
                chunks.append(current)
            current = p
        else:
            current = current + "\n" + p if current else p
    if current:
        chunks.append(current)
    return chunks


def _build_headers(env: dict) -> dict:
    api_key = env.get("VOLCENGINE_API_KEY", "")
    if api_key:
        logger.info("using X-Api-Key auth")
        return {
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": RESOURCE_ID,
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
    app_id = env.get("VOLCENGINE_APP_ID", "")
    access_token = env.get("VOLCENGINE_ACCESS_TOKEN", "")
    if app_id and access_token:
        logger.info("using X-Api-App-Key + X-Api-Access-Key auth")
        return {
            "X-Api-App-Key": app_id,
            "X-Api-Access-Key": access_token,
            "X-Api-Resource-Id": RESOURCE_ID,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
    raise RuntimeError("no auth credentials found in .env (need VOLCENGINE_API_KEY or VOLCENGINE_APP_ID+VOLCENGINE_ACCESS_TOKEN)")


async def synthesize(text: str, headers: dict, output_path: str,
                    voice: str = DEFAULT_VOICE,
                    speech_rate: int = 0,
                    loudness_rate: int = 0,
                    disable_markdown_filter: bool = True,
                    latex_parser: str = ""):

    logger.info(f"connecting to {ENDPOINT} ...")
    ws = await websockets.connect(ENDPOINT, additional_headers=headers, max_size=10 * 1024 * 1024)
    logid = ws.response.headers.get("x-tt-logid", "N/A")
    logger.info(f"connected, logid={logid}")

    all_audio = bytearray()
    try:
        await ws.send(_build_frame(EventType.StartConnection, payload=b"{}"))
        while True:
            data = await ws.recv()
            mt, ev, _, pl = _parse_response(data)
            if mt == MsgType.FullServerResponse and ev == EventType.ConnectionStarted:
                break
            if mt == MsgType.Error or (mt == MsgType.FullServerResponse and ev == EventType.ConnectionFailed):
                raise RuntimeError(f"connection failed: {pl.decode('utf-8', 'ignore')}")
        logger.info("connection started")

        def _build_additions():
            d = {"disable_markdown_filter": disable_markdown_filter}
            if latex_parser:
                d["latex_parser"] = latex_parser
            return json.dumps(d)

        base_req_params = {
            "speaker": voice,
            "audio_params": {
                "format": AUDIO_FORMAT,
                "sample_rate": SAMPLE_RATE,
                "speech_rate": speech_rate,
                "loudness_rate": loudness_rate,
            },
            "additions": _build_additions(),
        }

        session_id = str(uuid.uuid4())
        session_req = {
            "user": {"uid": str(uuid.uuid4())},
            "namespace": "BidirectionalTTS",
            "event": EventType.StartSession,
            "req_params": {**base_req_params},
        }
        await ws.send(_build_frame(EventType.StartSession, session_id, json.dumps(session_req).encode()))
        while True:
            data = await ws.recv()
            mt, ev, _, pl = _parse_response(data)
            if mt == MsgType.FullServerResponse and ev == EventType.SessionStarted:
                break
            if mt == MsgType.Error or (mt == MsgType.FullServerResponse and ev == EventType.SessionFailed):
                raise RuntimeError(f"session start failed: {pl.decode('utf-8', 'ignore')}")
        logger.info("session started, sending text character-by-character")
        logger.info(f"sending {len(text)} chars in single session")

        async def send_all_text():
            for ch in text:
                task_req = {
                    "user": {"uid": str(uuid.uuid4())},
                    "namespace": "BidirectionalTTS",
                    "event": EventType.TaskRequest,
                    "req_params": {
                        **base_req_params,
                        "text": ch,
                    },
                }
                await ws.send(_build_frame(EventType.TaskRequest, session_id, json.dumps(task_req).encode()))
                await asyncio.sleep(0.005)
            await ws.send(_build_frame(EventType.FinishSession, session_id, b"{}"))

        send_task = asyncio.create_task(send_all_text())

        while True:
            data = await ws.recv()
            mt, ev, _, pl = _parse_response(data)
            if mt == MsgType.AudioOnlyServer:
                all_audio.extend(pl)
            elif mt == MsgType.FullServerResponse:
                if ev == EventType.SessionFinished:
                    break
                elif ev == EventType.SessionFailed:
                    raise RuntimeError(f"session failed: {pl.decode('utf-8', 'ignore')}")
            elif mt == MsgType.Error:
                raise RuntimeError(f"error: {pl.decode('utf-8', 'ignore')}")

        await send_task
        logger.info(f"session done, total audio={len(all_audio)} bytes")

    finally:
        await ws.send(_build_frame(EventType.FinishConnection, payload=b"{}"))
        try:
            await asyncio.wait_for(ws.recv(), timeout=5)
        except Exception:
            pass
        await ws.close()
        logger.info("connection closed")

    output = Path(output_path)
    output.write_bytes(bytes(all_audio))
    logger.info(f"saved {len(all_audio)} bytes to {output}")
    return str(output)


def main():
    parser = argparse.ArgumentParser(description="火山引擎 TTS - Markdown 转语音")
    parser.add_argument("file", nargs="?", help="Markdown 文件路径")
    parser.add_argument("--text", help="直接输入文本")
    parser.add_argument("--output", default="output.mp3", help="输出文件路径")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="音色ID (默认 Vivi 2.0)")
    parser.add_argument("--speech-rate", type=int, default=20, help="语速 [-50,100], 默认20(1.2倍速)")
    parser.add_argument("--loudness-rate", type=int, default=0, help="音量 [-50,100], 默认0")
    parser.add_argument("--enable-markdown-filter", action="store_true", help="启用markdown过滤(默认不过滤)")
    parser.add_argument("--latex-parser", default="v2", help="LaTeX公式播报, 默认v2")
    args = parser.parse_args()

    env = load_env()
    try:
        headers = _build_headers(env)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.text:
        text = args.text
    elif args.file:
        text = read_markdown(args.file)
        logger.info(f"read {len(text)} chars from {args.file}")
    else:
        parser.print_help()
        sys.exit(1)

    if not text.strip():
        print("error: empty text", file=sys.stderr)
        sys.exit(1)

    asyncio.run(synthesize(
        text, headers, args.output,
        voice=args.voice,
        speech_rate=args.speech_rate,
        loudness_rate=args.loudness_rate,
        disable_markdown_filter=not args.enable_markdown_filter,
        latex_parser=args.latex_parser,
    ))


if __name__ == "__main__":
    main()
