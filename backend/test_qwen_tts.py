#!/usr/bin/env python3
"""
Qwen3-TTS FastAPI 集成测试脚本

测试 Qwen3-TTS 的各个功能：
1. 健康检查
2. 获取语音列表
3. CustomVoice 模式生成
4. 多语言测试
5. 不同说话人测试
"""

import requests
import time
from pathlib import Path
from typing import Optional


class QwenTTSClient:
    """Qwen3-TTS API 客户端"""

    def __init__(self, base_url: str = "http://localhost:8000", token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.headers = {}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def _request(self, method: str, endpoint: str, **kwargs):
        """发送请求"""
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, headers=self.headers, **kwargs)
        return response

    def health_check(self) -> dict:
        """健康检查"""
        response = self._request("GET", "/health")
        return response.json()

    def qwen_health(self) -> dict:
        """Qwen 模型健康检查"""
        response = self._request("GET", "/api/v1/tts/health/qwen")
        return response.json()

    def get_voices(self, engine: str = "qwen_tts") -> list:
        """获取语音列表"""
        response = self._request("GET", f"/api/v1/tts/voices?engine={engine}")
        return response.json()

    def generate_custom_voice(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> bytes:
        """生成语音 (CustomVoice 模式)"""
        payload = {
            "text": text,
            "speaker": speaker,
            "language": language,
        }
        if instruct:
            payload["instruct"] = instruct

        response = self._request("POST", "/api/v1/tts/qwen_tts/generate", json=payload)

        if response.status_code == 200:
            audio = response.content
            if output_path:
                Path(output_path).write_bytes(audio)
                print(f"✓ Audio saved to: {output_path}")
            return audio
        else:
            print(f"✗ Error: {response.status_code} - {response.text}")
            return b""

    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str = "Chinese",
        output_path: Optional[str] = None,
    ) -> bytes:
        """生成语音 (VoiceDesign 模式)"""
        payload = {
            "text": text,
            "instruct": instruct,
            "language": language,
        }

        response = self._request("POST", "/api/v1/tts/qwen_tts/design", json=payload)

        if response.status_code == 200:
            audio = response.content
            if output_path:
                Path(output_path).write_bytes(audio)
                print(f"✓ Audio saved to: {output_path}")
            return audio
        else:
            print(f"✗ Error: {response.status_code} - {response.text}")
            return b""

    def stream_unified(
        self,
        text: str,
        engine: str = "qwen_tts",
        speaker: str = "Vivian",
        language: str = "Chinese",
        output_path: Optional[str] = None,
    ) -> bytes:
        """统一流式端点"""
        payload = {
            "text": text,
            "engine": engine,
            "speaker": speaker,
            "language": language,
        }

        response = self._request("POST", "/api/v1/tts/stream", json=payload)

        if response.status_code == 200:
            audio = response.content
            if output_path:
                Path(output_path).write_bytes(audio)
                print(f"✓ Audio saved to: {output_path}")
            return audio
        else:
            print(f"✗ Error: {response.status_code} - {response.text}")
            return b""


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_health(client: QwenTTSClient):
    """测试 1: 健康检查"""
    print_section("Test 1: Health Check")

    health = client.health_check()
    print(f"API Status: {health.get('status')}")
    print(f"Engines: {health.get('engines')}")
    print(f"Qwen Enabled: {health.get('qwen_enabled')}")

    qwen_health = client.qwen_health()
    print(f"\nQwen Model Status: {qwen_health.get('status')}")
    model_info = qwen_health.get('model', {})
    print(f"Model Type: {model_info.get('model_type')}")
    print(f"Model Size: {model_info.get('model_size')}")
    print(f"Device: {model_info.get('device')}")
    print(f"Initialized: {model_info.get('initialized')}")


def test_voices(client: QwenTTSClient):
    """测试 2: 获取语音列表"""
    print_section("Test 2: Get Voice List")

    voices = client.get_voices("qwen_tts")
    print(f"Total speakers: {len(voices)}\n")

    for voice in voices:
        print(f"  - {voice['id']}: {voice.get('description', 'N/A')}")


def test_basic_generation(client: QwenTTSClient):
    """测试 3: 基本语音生成"""
    print_section("Test 3: Basic Speech Generation")

    text = "你好，这是 Qwen3-TTS 的测试语音。"
    print(f"Text: {text}")
    print(f"Speaker: Vivian")
    print(f"Language: Chinese\n")

    start = time.time()
    audio = client.generate_custom_voice(
        text=text,
        speaker="Vivian",
        language="Chinese",
        output_path="test_vivian.wav"
    )
    elapsed = time.time() - start

    if audio:
        size_kb = len(audio) / 1024
        print(f"✓ Generated {size_kb:.1f}KB audio in {elapsed:.2f}s")


def test_multilingual(client: QwenTTSClient):
    """测试 4: 多语言测试"""
    print_section("Test 4: Multilingual Test")

    tests = [
        ("Chinese", "Vivian", "你好，欢迎使用 Qwen3-TTS 语音合成系统。"),
        ("English", "Ryan", "Hello! Welcome to the Qwen3-TTS speech synthesis system."),
        ("Japanese", "Ono_Anna", "こんにちは！Qwen3-TTS音声合成システムへようこそ。"),
        ("Korean", "Sohee", "안녕하세요! Qwen3-TTS 음성 합성 시스템에 오신 것을 환영합니다."),
    ]

    for lang, speaker, text in tests:
        print(f"\n[{lang}] {speaker}")
        print(f"  Text: {text[:50]}...")

        start = time.time()
        audio = client.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=lang,
            output_path=f"test_{lang.lower()}_{speaker}.wav"
        )
        elapsed = time.time() - start

        if audio:
            print(f"  ✓ Generated in {elapsed:.2f}s")


def test_different_speakers(client: QwenTTSClient):
    """测试 5: 不同说话人测试"""
    print_section("Test 5: Different Speakers Test")

    speakers = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"]

    for speaker in speakers:
        text = f"你好，我是{speaker}，很高兴认识你。"
        print(f"\n[{speaker}]")
        print(f"  Text: {text}")

        start = time.time()
        audio = client.generate_custom_voice(
            text=text,
            speaker=speaker,
            language="Chinese",
            output_path=f"test_speaker_{speaker}.wav"
        )
        elapsed = time.time() - start

        if audio:
            print(f"  ✓ Generated in {elapsed:.2f}s")


def test_style_instruction(client: QwenTTSClient):
    """测试 6: 风格指令测试"""
    print_section("Test 6: Style Instruction Test")

    text = "今天天气真好，适合出去散步。"
    instructions = [
        None,
        "用温柔的声音说",
        "用兴奋的语气说",
        "用低沉的声音说",
    ]

    for i, instruct in enumerate(instructions, 1):
        print(f"\n[Test 6.{i}]")
        print(f"  Text: {text}")
        print(f"  Instruction: {instruct or '(默认)'}")

        start = time.time()
        output_path = f"test_style_{i}.wav"
        audio = client.generate_custom_voice(
            text=text,
            speaker="Vivian",
            language="Chinese",
            instruct=instruct,
            output_path=output_path
        )
        elapsed = time.time() - start

        if audio:
            print(f"  ✓ Generated in {elapsed:.2f}s")


def test_unified_stream(client: QwenTTSClient):
    """测试 7: 统一流式端点"""
    print_section("Test 7: Unified Stream Endpoint")

    text = "这是使用统一流式端点生成的语音。"
    print(f"Text: {text}")
    print(f"Engine: qwen_tts")

    start = time.time()
    audio = client.stream_unified(
        text=text,
        engine="qwen_tts",
        speaker="Vivian",
        language="Chinese",
        output_path="test_unified.wav"
    )
    elapsed = time.time() - start

    if audio:
        size_kb = len(audio) / 1024
        print(f"✓ Generated {size_kb:.1f}KB audio in {elapsed:.2f}s")


def test_long_text(client: QwenTTSClient):
    """测试 8: 长文本测试"""
    print_section("Test 8: Long Text Test")

    text = """这是一个较长的文本测试。在实际应用场景中，我们经常需要处理各种长度的文本输入。
Qwen3-TTS 模型通过其先进的架构设计，能够高效地处理这些不同长度的文本，并生成高质量的语音输出。
同时，模型还支持流式生成，可以在文本输入的同时就开始输出语音，大大降低了端到端的延迟。
这个测试可以验证模型在处理较长文本时的稳定性和性能表现。"""

    print(f"Text length: {len(text)} characters")
    print(f"Text preview: {text[:50]}...")

    start = time.time()
    audio = client.generate_custom_voice(
        text=text,
        speaker="Vivian",
        language="Chinese",
        output_path="test_long.wav"
    )
    elapsed = time.time() - start

    if audio:
        size_kb = len(audio) / 1024
        duration = size_kb / 200  # 粗略估计 (WAV ~200KB/s)
        print(f"✓ Generated {size_kb:.1f}KB audio in {elapsed:.2f}s")
        print(f"  Estimated duration: ~{duration:.1f}s")
        print(f"  RTF (Real-Time Factor): {elapsed/duration:.2f}x")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("  Qwen3-TTS FastAPI 集成测试")
    print("=" * 60)

    # 从环境变量或命令行获取 token
    import os
    token = os.getenv("TTS_TOKEN") or os.getenv("ADMIN_TOKEN")

    if not token:
        print("\n⚠ Warning: No token found!")
        print("  Set TTS_TOKEN or ADMIN_TOKEN environment variable")
        print("  Or the server might be running with localhost bypass\n")
        response = input("Continue without token? (y/n): ")
        if response.lower() != 'y':
            return

    # 初始化客户端
    base_url = os.getenv("TTS_URL", "http://localhost:8000")
    client = QwenTTSClient(base_url=base_url, token=token)

    try:
        # 运行测试
        test_health(client)
        test_voices(client)
        test_basic_generation(client)
        test_multilingual(client)
        test_different_speakers(client)
        test_style_instruction(client)
        test_unified_stream(client)
        test_long_text(client)

        # 完成
        print_section("All Tests Completed")
        print("✓ Check the generated WAV files in the current directory")
        print("  Files: test_*.wav")

    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Cannot connect to server")
        print(f"  Make sure the server is running at: {base_url}")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()
