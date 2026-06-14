from __future__ import annotations

from typing import Any

from app.services.base import TTSEngine


class EngineRegistry:
    """Engine 注册表。支持运行时注册和创建。"""

    _engines: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：注册一个 engine 类。

        Usage:
            @EngineRegistry.register("qwen")
            class Qwen3TTSEngine: ...
        """
        def decorator(engine_cls: type) -> type:
            cls._engines[name] = engine_cls
            return engine_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> TTSEngine:
        """创建 engine 实例。

        Raises:
            ValueError: 未知 engine name
        """
        if name not in cls._engines:
            available = ", ".join(cls._engines.keys())
            raise ValueError(f"Unknown engine: '{name}'. Available: [{available}]")
        return cls._engines[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """返回已注册的 engine 名称列表。"""
        return list(cls._engines.keys())


def register_builtin_engines() -> None:
    """注册所有内置 engine。在 app 启动时调用。"""
    from app.services.edge_engine import EdgeTTSEngine
    from app.services.volcengine_engine import VolcengineTTSEngine

    EngineRegistry.register("edge")(EdgeTTSEngine)
    EngineRegistry.register("volcengine")(VolcengineTTSEngine)

    try:
        from app.services.qwen_engine import Qwen3TTSEngine
        EngineRegistry.register("qwen")(Qwen3TTSEngine)
    except ImportError:
        pass
