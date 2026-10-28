"""Chroma 遥测的空实现，避免本地索引时输出 PostHog 兼容性错误。"""

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetryClient(ProductTelemetryClient):
    """忽略 Chroma 产品遥测事件。"""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        """丢弃遥测事件。"""
        return None
