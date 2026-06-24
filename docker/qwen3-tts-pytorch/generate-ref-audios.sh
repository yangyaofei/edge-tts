#!/bin/bash
# Generate reference audio files for TTS voice cloning.
# Uses edge-tts (npm) to create high-quality Chinese speech samples.
#
# Usage: bash docker/qwen3-tts-pytorch/generate-ref-audios.sh
# Output: docker/qwen3-tts-pytorch/ref_audios/{ref_zh,ref_zh_male}.{wav,txt}

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/ref_audios"
TEXT="人工智能技术在过去几年里取得了巨大的进步，尤其是在自然语言处理和大语言模型方面。"

mkdir -p "$OUT_DIR"

echo "=== Generating reference audio files ==="
echo "Text: $TEXT"
echo "Output: $OUT_DIR"
echo ""

# Female voice (XiaoxiaoNeural)
echo "[1/2] Generating female voice (zh-CN-XiaoxiaoNeural)..."
edge-tts --voice zh-CN-XiaoxiaoNeural --text "$TEXT" --write-media "$OUT_DIR/ref_zh.mp3"
ffmpeg -y -i "$OUT_DIR/ref_zh.mp3" -ar 24000 -ac 1 -f wav "$OUT_DIR/ref_zh.wav" 2>/dev/null
rm "$OUT_DIR/ref_zh.mp3"
echo -n "$TEXT" > "$OUT_DIR/ref_zh.txt"
echo "  -> ref_zh.wav + ref_zh.txt"

# Male voice (YunxiNeural)
echo "[2/2] Generating male voice (zh-CN-YunxiNeural)..."
edge-tts --voice zh-CN-YunxiNeural --text "$TEXT" --write-media "$OUT_DIR/ref_zh_male.mp3"
ffmpeg -y -i "$OUT_DIR/ref_zh_male.mp3" -ar 24000 -ac 1 -f wav "$OUT_DIR/ref_zh_male.wav" 2>/dev/null
rm "$OUT_DIR/ref_zh_male.mp3"
echo -n "$TEXT" > "$OUT_DIR/ref_zh_male.txt"
echo "  -> ref_zh_male.wav + ref_zh_male.txt"

echo ""
echo "=== Done ==="
ls -la "$OUT_DIR/"
