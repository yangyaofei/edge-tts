# 实施指南 - 第一阶段

本文件详细说明了 **后端 (Backend)** 和 **Chrome 插件 (Extension)** 的具体实施步骤。

## 第一部分：Python 后端 (FastAPI)

### 1.1 项目结构
```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # 入口文件，处理 CORS, 应用定义
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           ├── tts.py   # TTS 路由 (Edge, 之后加入 Qwen)
│   │           └── text.py  # 文本处理 (分块逻辑)
│   ├── core/
│   │   └── config.py        # 环境变量、配置
│   ├── services/
│   │   ├── edge_engine.py   # edge-tts 库的封装
│   │   └── chunker.py       # 文本切分逻辑
│   └── schemas/
│       └── tts.py           # Pydantic 数据模型
├── tests/
├── .env
└── requirements.txt
```

### 1.2 核心依赖 (`requirements.txt`)
- `fastapi`
- `uvicorn[standard]`
- `edge-tts` (核心 TTS 引擎)
- `pydantic-settings`
- `python-multipart`
- `aiofiles`

### 1.3 实现步骤

**第一步：Edge-TTS 服务 (`services/edge_engine.py`)**
- 实现异步函数 `generate_audio_stream(text, voice, rate, pitch)`。
- 内部调用 `edge_tts.Communicate(text, voice).stream()`。
- 使用 `yield` 产生字节块，供 FastAPI 的 `StreamingResponse` 使用。

**第二步：API 路由 (`api/v1/endpoints/tts.py`)**
- 接口：`POST /api/v1/tts/edge/stream`。
- 返回：`StreamingResponse(generator, media_type="audio/mpeg")`。

**第三步：文本分块器 (`services/chunker.py`)**
- 基础策略：按换行符 `\n` 切分。
- 进阶策略：如果段落过长（>500字），按标点符号（`.` `!` `?` `。` `！` `？`）切分。

---

## 第二部分：Chrome 插件 (Vue 3 + Vite + CRXJS)

### 2.1 项目结构
```text
extension/
├── src/
│   ├── assets/
│   ├── components/
│   ├── background/
│   │   └── index.ts      # Service Worker (后台脚本)
│   ├── content/
│   │   └── index.ts      # 文本提取逻辑 (注入页面)
│   ├── sidepanel/
│   │   ├── index.html
│   │   └── SidePanel.vue # 主 UI 界面
│   ├── offscreen/
│   │   ├── index.html
│   │   └── audio-player.ts # 实际处理音频播放的隐藏页面
│   └── manifest.json
vite.config.ts
└── package.json
```

### 2.2 关键技术
- **构建工具:** `vite` 配合 `@crxjs/vite-plugin`。
- **状态管理:** `pinia` (管理播放队列、当前播放索引、播放状态)。
- **UI 框架:** `naive-ui` 或 `ant-design-vue`。

### 2.3 实现步骤

**第一步：Manifest 配置 (`manifest.json`)**
- 权限：`"sidePanel"`, `"activeTab"`, `"scripting"`, `"offscreen"`, `"storage"`。
- 主机权限：`http://localhost:8000/*` (允许跨域访问后端)。

**第二步：Offscreen Document (音频播放器)**
- **为什么？** 因为 Manifest V3 的 Service Worker 无法持久播放音频，且没有 DOM 权限操作 `<audio>`。
- **逻辑：**
    - 监听来自 SidePanel 的消息。
    - 接收音频 Blob URL 或流地址。
    - 管理 `<audio>` 标签。
    - 向 SidePanel 回传 `AUDIO_ENDED` (播放结束) 或 `AUDIO_TIMEUPDATE` (进度更新) 消息。

**第三步：SidePanel UI (控制器)**
- **状态：** `isPlaying` (播放中), `currentChunkIndex` (当前段落), `chunks[]` (所有段落)。
- **播放循环逻辑：**
    1. 用户点击“朗读页面”。
    2. SidePanel 通知 Content Script 提取文本。
    3. SidePanel 调用后端 `/text/chunk` 获取分块。
    4. **循环：**
        - 获取 `chunks[i]`。
        - 调用后端 `/tts/edge/stream`。
        - 将结果转化为 Blob URL 发送给 **Offscreen Document** 播放。
        - 等待 Offscreen 返回 `AUDIO_ENDED`。
        - 自动开始播放 `chunks[i+1]`。

**第四步：Content Script (阅读器)**
- 提取逻辑：优先寻找 `article` 标签，或使用类似 `@mozilla/readability` 的逻辑过滤导航栏和广告。

---

## 第三部分：开发流与测试

1.  **启动后端:** `uvicorn app.main:app --reload --port 8000`
2.  **构建插件:** `npm run dev` (Vite 监听模式)。
3.  **加载插件:** 在 Chrome `chrome://extensions` 中开启“开发者模式”，选择“加载已解压的扩展程序”，指向 `extension/dist`。
4.  **测试环节:** 
    - 打开一个新闻网页。
    - 点击插件图标打开侧边栏 (SidePanel)。
    - 点击播放。
    - 验证音频流是否正常、段落切换是否连贯。
