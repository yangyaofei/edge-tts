# Qwen3-TTS 超时问题解决方案

## 问题原因

Qwen3-TTS 在 MPS (Apple Silicon) 上的生成速度约为：
- **实时率（RTF）**: 0.2-0.3x（比实时慢 3-5 倍）
- **短文本**（10字）：~7 秒
- **中文本**（100字）：~86 秒
- **长文本**（300字）：~3 分钟

curl 默认超时时间较短，长文本生成会超时。

---

## 快速解决方案

### 1. 增加 curl 超时时间

```bash
curl -X POST "http://localhost:8700/api/v1/tts/qwen_tts/generate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"你的文本...","speaker":"Vivian","language":"Chinese"}' \
  --output output.wav \
  --max-time 300      # 5 分钟超时（关键！）
```

### 超时时间参考表

| 文本长度 | 预估生成时间 | 建议超时设置 |
|---------|------------|-------------|
| < 50 字 | < 30 秒 | `--max-time 60` |
| 50-100 字 | 30-90 秒 | `--max-time 120` |
| 100-200 字 | 90-180 秒 | `--max-time 300` |
| 200-300 字 | 3-5 分钟 | `--max-time 600` |
| > 300 字 | > 5 分钟 | **建议分段** |

---

## 推荐的 curl 命令模板

### 短文本（< 50 字）

```bash
curl -X POST "http://localhost:8700/api/v1/tts/qwen_tts/generate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"你好世界","speaker":"Vivian","language":"Chinese"}' \
  --output short.wav \
  --max-time 60
```

### 中文本（50-150 字）

```bash
curl -X POST "http://localhost:8700/api/v1/tts/qwen_tts/generate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"这是一段中等长度的文本，大约一百个字左右。","speaker":"Vivian","language":"Chinese","max_new_tokens":2048}' \
  --output medium.wav \
  --max-time 180
```

### 长文本（> 150 字）

```bash
curl -X POST "http://localhost:8700/api/v1/tts/qwen_tts/generate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"长文本...","speaker":"Vivian","language":"Chinese","max_new_tokens":4096}' \
  --output long.wav \
  --max-time 600 \
  --connect-timeout 60
```

---

## Python 客户端示例（自动处理超时）

```python
import requests
import time

TOKEN = "your_token_here"
BASE_URL = "http://localhost:8700"

def generate_tts(text, output_file, timeout=600):
    """生成 TTS，支持长文本"""
    headers = {"Authorization": f"Bearer {TOKEN}"}
    payload = {
        "text": text,
        "speaker": "Vivian",
        "language": "Chinese",
        "max_new_tokens": 4096  # 长文本需要更多 tokens
    }

    print(f"开始生成: {len(text)} 字符")
    start = time.time()

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/tts/qwen_tts/generate",
            headers=headers,
            json=payload,
            timeout=timeout  # 10 分钟超时
        )

        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            elapsed = time.time() - start
            print(f"✓ 完成! 耗时: {elapsed:.1f}秒, 文件: {output_file}")
            return True
        else:
            print(f"✗ 错误: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"✗ 超时! 生成时间超过 {timeout} 秒")
        return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False

# 使用示例
generate_tts(
    "这是一段较长的文本内容...",
    "output.wav",
    timeout=600  # 10 分钟
)
```

---

## 长文本最佳实践：分段处理

对于超长文本（> 200 字），**强烈建议分段处理**：

```python
import requests
import re

TOKEN = "your_token_here"
BASE_URL = "http://localhost:8700"

def split_text(text, max_length=150):
    """按标点符号分段"""
    # 按句子分割
    sentences = re.split(r'[。！？.!?]', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + "。"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + "。"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def generate_tts_long(text, output_prefix):
    """分段生成并拼接"""
    chunks = split_text(text, max_length=150)
    print(f"文本分为 {len(chunks)} 段")

    audio_files = []
    for i, chunk in enumerate(chunks, 1):
        output_file = f"{output_prefix}_{i}.wav"
        print(f"\n[{i}/{len(chunks)}] 生成: {chunk[:30]}...")

        # 调用 API
        headers = {"Authorization": f"Bearer {TOKEN}"}
        response = requests.post(
            f"{BASE_URL}/api/v1/tts/qwen_tts/generate",
            headers=headers,
            json={
                "text": chunk,
                "speaker": "Vivian",
                "language": "Chinese"
            },
            timeout=180  # 每段 3 分钟
        )

        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            audio_files.append(output_file)
            print(f"✓ 第 {i} 段完成")
        else:
            print(f"✗ 第 {i} 段失败")

    print(f"\n完成！生成了 {len(audio_files)} 个音频文件")
    return audio_files

# 使用示例
long_text = """第一段内容... 第二段内容... 第三段内容..."""
files = generate_tts_long(long_text, "chapter1")

# 然后可以用 ffmpeg 或其他工具拼接音频
# ffmpeg -f concat -i filelist.txt -c copy output.wav
```

---

## 常见错误及解决

### 错误 1: `curl: (28) Operation timed out`

**原因**: 超时时间太短

**解决**:
```bash
# 增加 --max-time 参数
curl ... --max-time 300
```

### 错误 2: `curl: (52) Empty reply from server`

**原因**: 服务器可能在生成过程中断开连接

**解决**:
```bash
curl ... --keep-alive --max-time 600
```

### 错误 3: 音频被截断

**原因**: `max_new_tokens` 不够

**解决**:
```bash
curl ... -d '{"text":"...","max_new_tokens":8192}'
```

---

## 总结

| 问题 | 解决方案 |
|-----|---------|
| curl 超时 | 添加 `--max-time 600` |
| 音频截断 | 增加 `max_new_tokens` |
| 生成太慢 | 分段处理文本 |
| 频繁超时 | 使用 Python 客户端，自动重试 |

**最佳实践**: 对于 > 150 字的文本，使用分段处理！
