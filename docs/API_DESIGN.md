# API 接口设计文档

## 1. 基础信息
- **Base URL:** `http://localhost:8000/api/v1`
- **格式:** JSON
- **鉴权:** 暂无 (本地运行)，后续可添加 API Key。

## 2. 文本处理 (Text Processing)

### 2.1 文本分块
将长文本拆分为适合 TTS 生成的短句/段落。

- **Endpoint:** `POST /text/chunk`
- **Request Body:**
  ```json
  {
    "text": "长文本内容...",
    "strategy": "paragraph" // 可选: paragraph, sentence
  }
  ```
- **Response:**
  ```json
  {
    "chunks": [
      { "id": 0, "text": "第一段文本..." },
      { "id": 1, "text": "第二段文本..." }
    ]
  }
  ```

## 3. 语音合成 (TTS)

### 3.1 Edge-TTS 流式生成
使用微软 Edge 在线语音生成音频流。

- **Endpoint:** `POST /tts/edge/stream`
- **Request Body:**
  ```json
  {
    "text": "需要朗读的文本",
    "voice": "zh-CN-XiaoxiaoNeural", // 默认 voice
    "rate": "+0%", // 语速
    "pitch": "+0Hz" // 音调
  }
  ```
- **Response:** Audio Stream (`audio/mpeg`)
  - 使用 Chunked Transfer Encoding 返回二进制音频数据。

### 3.2 Qwen-TTS 流式生成 (v2)
使用阿里 Dashscope API 生成。

- **Endpoint:** `POST /tts/qwen/stream`
- **Request Body:**
  ```json
  {
    "text": "需要朗读的文本",
    "voice": "cherry", // Qwen voice ID
    "format": "mp3"
  }
  ```
- **Response:** Audio Stream (`audio/mpeg`)

### 3.3 获取可用语音
获取支持的语音列表。

- **Endpoint:** `GET /tts/voices`
- **Query Params:**
  - `engine`: `edge` | `qwen` (默认 `edge`)
- **Response:**
  ```json
  [
    { "id": "zh-CN-XiaoxiaoNeural", "name": "Xiaoxiao (Chinese)", "engine": "edge" },
    { "id": "en-US-AriaNeural", "name": "Aria (English)", "engine": "edge" }
  ]
  ```

## 4. 前端交互流程 (Chrome Ext)

1.  **提取:** Content Script 获取网页正文。
2.  **分块:** 调用 `/text/chunk` 获得段落列表。
3.  **播放:** 前端维护播放队列，依次对当前段落调用 `/tts/edge/stream`。
    - *优化:* 预加载（Preload）下一段落的音频，保证播放流畅。
