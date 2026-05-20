# TTS 验证脚本需求

## 原始需求

根据 https://www.volcengine.com/docs/6561/1329505?lang=zh (WebSocket 双向流式 API) 文档，写一个简单的 Python 脚本验证火山引擎 TTS 效果。

## 测试文件

/Users/yangyaofei/workspace/lboskj/agent-service/docs/acp-sdk/PLAN.md

## API 认证信息（两种方式）

### 方式 A: X-Api-Key（新控制台 API Key）
- `X-Api-Key`: `ark-f77015fe-8789-49bc-bf4c-11e35d74471e-d29c6`
- `X-Api-Resource-Id`: `seed-tts-2.0`
- `X-Api-Request-Id`: 随机 UUID

### 方式 B: X-Api-App-Key + X-Api-Access-Key（旧控制台）
- `X-Api-App-Key` (APP ID): `7497430716`
- `X-Api-Access-Key` (Access Token): `7BYWv6ZObYtf5065rFsM0eUo-ec1Dio8`
- `X-Api-Resource-Id`: `seed-tts-2.0`

## .env 文件

```
VOLCENGINE_API_KEY=ark-f77015fe-8789-49bc-bf4c-11e35d74471e-d29c6
VOLCENGINE_APP_ID=7497430716
VOLCENGINE_ACCESS_TOKEN=7BYWv6ZObYtf5065rFsM0eUo-ec1Dio8
```

## 技术选型

- 使用 uv 管理 Python 环境
- 基于官方文档的双向流式 WebSocket API (wss://openspeech.bytedance.com/api/v3/tts/bidirection)
- 使用官方 Demo 下载的协议层代码 (tmp/volcengine_bidirection_demo)

## 音色参考

https://www.volcengine.com/docs/6561/1257544?lang=zh#豆包语音合成模型2-0音色列表

## 当前进展

1. ✅ 已获取文档内容（双向流式 WebSocket 二进制协议）
2. ✅ 已下载官方 Python Demo (tmp/volcengine_bidirection_demo/)
3. ✅ 已读取测试文件 PLAN.md
4. 🔄 需要更新 .env 和脚本，使用两种认证方式 fallback
