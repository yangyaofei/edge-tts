#!/usr/bin/env python3
"""
简单的 Qwen3-TTS 客户端，自动处理超时问题
"""

import requests
import sys
import time

# 配置
BASE_URL = "http://localhost:8700"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImFkbWluIjp0cnVlfQ.7VntQA6Wqbpj6mQePhqknZiMvQStIp5BbnS2zfcGnc4"

def generate_tts(text, speaker="Vivian", language="Chinese", output_file="output.wav", timeout=600):
    """
    生成 TTS 语音

    Args:
        text: 要转换的文本
        speaker: 说话人 (默认 Vivian)
        language: 语言 (默认 Chinese)
        output_file: 输出文件名
        timeout: 超时时间（秒），默认 10 分钟
    """
    headers = {"Authorization": f"Bearer {TOKEN}"}
    payload = {
        "text": text,
        "speaker": speaker,
        "language": language,
        "max_new_tokens": 4096 if len(text) > 100 else 2048
    }

    print(f"🎤 开始生成语音...")
    print(f"   文本长度: {len(text)} 字符")
    print(f"   说话人: {speaker}")
    print(f"   语言: {language}")
    print(f"   超时设置: {timeout} 秒")
    print()

    start = time.time()

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/tts/qwen_tts/generate",
            headers=headers,
            json=payload,
            timeout=timeout
        )

        elapsed = time.time() - start

        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)

            file_size = len(response.content) / 1024
            print(f"✅ 成功!")
            print(f"   耗时: {elapsed:.1f} 秒")
            print(f"   文件: {output_file} ({file_size:.1f} KB)")
            return True
        else:
            print(f"❌ 错误: HTTP {response.status_code}")
            print(f"   {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"❌ 超时! 生成时间超过 {timeout} 秒")
        print(f"   建议:")
        print(f"   1. 缩短文本长度")
        print(f"   2. 增加超时时间: python qwen_tts_client.py --timeout 1200")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("用法: python qwen_tts_client.py '文本内容' [输出文件]")
        print()
        print("示例:")
        print('  python qwen_tts_client.py "你好世界"')
        print('  python qwen_tts_client.py "长文本..." output.wav')
        print()
        print("选项:")
        print('  --speaker NAME    说话人 (Vivian, Serena, Uncle_Fu, 等)')
        print('  --language LANG    语言 (Chinese, English, Japanese, 等)')
        print('  --timeout SEC     超时时间（秒）')
        return

    # 解析参数
    text = sys.argv[1]
    output_file = "output.wav"
    speaker = "Vivian"
    language = "Chinese"
    timeout = 600

    # 解析可选参数
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--speaker":
            speaker = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "--language":
            language = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "--timeout":
            timeout = int(sys.argv[i+1])
            i += 2
        else:
            output_file = sys.argv[i]
            i += 1

    # 生成语音
    generate_tts(text, speaker, language, output_file, timeout)


if __name__ == "__main__":
    main()
