"""
文本处理 API 端点模块

本模块提供文本处理相关的 API 端点，包括：
- 文本分块（Text Chunking）：将长文本分割成适合 TTS 的短片段

功能特点：
- 支持多种分块策略（段落、句子、标点）
- 返回带编号的文本块
- 自动处理空白字符
"""

from fastapi import APIRouter, Depends
from app.schemas.tts import TextChunkRequest, TextChunkResponse, Chunk
from app.services.chunker import TextChunker
from app.core.security import verify_token

router = APIRouter()


@router.post("/chunk", response_model=TextChunkResponse, dependencies=[Depends(verify_token)])
async def chunk_text(request: TextChunkRequest):
    """
    将文本分割成多个块

    这个端点将长文本分割成适合 TTS 处理的短片段。
    对于 Qwen3-TTS 等模型，建议将文本限制在 150 字以内以获得最佳效果。

    Args:
        request: 分块请求
            - text: 要分块的文本
            - strategy: 分块策略
                - "paragraph": 按段落分割（默认，推荐）
                - "sentence": 按句子分割（未实现）
                - "punctuation": 按标点符号分割（未实现）

    Returns:
        TextChunkResponse: 分块结果
            - chunks: 文本块列表，每个块包含：
                - id: 块编号（从 0 开始）
                - text: 文本内容

    Example:
        请求:
        ```json
        {
            "text": "第一段。\\n\\n第二段。\\n\\n第三段。",
            "strategy": "paragraph"
        }
        ```

        响应:
        ```json
        {
            "chunks": [
                {"id": 0, "text": "第一段。"},
                {"id": 1, "text": "第二段。"},
                {"id": 2, "text": "第三段。"}
            ]
        }
        ```

    Note:
        - 空文本会返回空列表
        - 连续的空行会被自动过滤
        - 每个文本块会自动去除首尾空白

    See Also:
        - Qwen TTS 文档建议长文本应分段处理
        - 使用分段可以避免超时问题
    """
    chunks_text = TextChunker.chunk_text(request.text, request.strategy)
    chunks = [Chunk(id=i, text=t) for i, t in enumerate(chunks_text)]
    return TextChunkResponse(chunks=chunks)
