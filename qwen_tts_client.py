#!/usr/bin/env python3
"""
Qwen3-TTS 命令行客户端

这是一个简单易用的客户端，用于调用 Qwen3-TTS API 生成语音。
自动处理超时问题，支持长文本生成。

用法:
    python qwen_tts_client.py "要转换的文本" [输出文件] [选项]

示例:
    # 基础用法
    python qwen_tts_client.py "你好世界"

    # 指定输出文件
    python qwen_tts_client.py "你好世界" output.wav

    # 选择说话人
    python qwen_tts_client.py "你好世界" --speaker Serena

    # 长文本（增加超时）
    python qwen_tts_client.py "长文本..." --timeout 1200
"""

import requests
import sys
import time

# ===================================================================
# 配置
# ===================================================================

# API 服务地址
BASE_URL = "http://localhost:8700"

# 访问令牌（管理员 Token，永不过期）
# 注意：生产环境应使用环境变量或配置文件管理
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImFkbWluIjp0cnVlfQ.7VntQA6Wqbpj6mQePhqknZiMvQStIp5BbnS2zfcGnc4"


# ===================================================================
# TTS 生成函数
# ===================================================================

def generate_tts(text, speaker="Vivian", language="Chinese", output_file="output.wav", timeout=600):
    """
    调用 Qwen3-TTS API 生成语音

    Args:
        text: 要转换的文本
        speaker: 说话人名称
            可选: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
        language: 语言
            可选: Auto, Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian
        output_file: 输出文件名
        timeout: 超时时间（秒），默认 10 分钟
            - 短文本（< 50 字）：建议 60 秒
            - 中文本（50-150 字）：建议 180 秒
            - 长文本（> 150 字）：建议 600 秒或更长

    Returns:
        bool: 成功返回 True，失败返回 False
    """
    # 构建请求头
    headers = {"Authorization": f"Bearer {TOKEN}"}

    # 构建请求体
    payload = {
        "text": text,
        "speaker": speaker,
        "language": language,
        "max_new_tokens": 4096 if len(text) > 100 else 2048  # 长文本需要更多 tokens
    }

    # 显示任务信息
    print(f"🎤 开始生成语音...")
    print(f"   文本长度: {len(text)} 字符")
    print(f"   说话人: {speaker}")
    print(f"   语言: {language}")
    print(f"   超时设置: {timeout} 秒")
    print()

    start = time.time()

    try:
        # 发送 POST 请求到 TTS API
        response = requests.post(
            f"{BASE_URL}/api/v1/tts/qwen_tts/generate",
            headers=headers,
            json=payload,
            timeout=timeout
        )

        elapsed = time.time() - start

        # 检查响应状态
        if response.status_code == 200:
            # 保存音频文件
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
        # 超时处理
        print(f"❌ 超时! 生成时间超过 {timeout} 秒")
        print(f"   建议:")
        print(f"   1. 缩短文本长度")
        print(f"   2. 增加超时时间: python qwen_tts_client.py --timeout 1200")
        return False
    except Exception as e:
        # 其他错误
        print(f"❌ 错误: {e}")
        return False


# ===================================================================
# 命令行入口
# ===================================================================


def main():
    """
    命令行入口函数

    解析命令行参数并调用 TTS 生成函数。
    """
    # 检查必需参数
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

    # 调用 TTS 生成函数
    generate_tts(text, speaker, language, output_file, timeout)


if __name__ == "__main__":
    main()
