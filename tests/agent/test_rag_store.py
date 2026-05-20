"""测试 RAG 存储的模型隔离行为。"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from nanobot.agent.rag.store import RAGStore


def test_default_embedding_uses_legacy_collection_and_registry(tmp_path) -> None:
    """默认 embedding 继续使用原 collection，避免破坏旧索引。"""
    store = RAGStore(tmp_path / "memory", embedding_model="all-MiniLM-L6-v2")

    assert store.collection_name == "rag_documents"
    assert store.registry_file == tmp_path / "memory" / "rag" / "document_registry.json"


def test_custom_embedding_uses_model_specific_collection_and_registry(tmp_path) -> None:
    """自定义 embedding 使用独立 collection，避免和旧配置冲突。"""
    store = RAGStore(tmp_path / "memory", embedding_model="Qwen/Qwen3-Embedding-8B")

    assert store.collection_name.startswith("rag_documents_qwen_qwen3-embedding-8b_")
    assert store.registry_file == tmp_path / "memory" / "rag" / f"{store.collection_name}_registry.json"


def test_different_embedding_models_do_not_share_collection(tmp_path) -> None:
    """不同 embedding 模型不能共用同一个向量 collection。"""
    qwen = RAGStore(tmp_path / "memory", embedding_model="Qwen/Qwen3-Embedding-8B")
    bge = RAGStore(tmp_path / "memory", embedding_model="BAAI/bge-m3")

    assert qwen.collection_name != bge.collection_name
    assert qwen.registry_file != bge.registry_file


def test_stats_exposes_active_collection_name(tmp_path) -> None:
    """状态信息里显示当前使用的 collection，便于排查索引问题。"""
    store = RAGStore(tmp_path / "memory", embedding_model="Qwen/Qwen3-Embedding-8B")
    store.count = lambda: 0  # type: ignore[method-assign]

    stats = store.get_stats()

    assert stats["collection_name"] == store.collection_name


def test_ensure_collection_uses_model_specific_name(monkeypatch, tmp_path) -> None:
    """初始化 Chroma 时使用隔离后的 collection 名称。"""
    calls: list[dict] = []
    fake_collection = object()

    class FakeClient:
        def __init__(self, path: str):
            self.path = path

        def get_or_create_collection(self, **kwargs):
            calls.append(kwargs)
            return fake_collection

    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=FakeClient))
    store = RAGStore(tmp_path / "memory", embedding_model="Qwen/Qwen3-Embedding-8B")
    store._create_embedding_function = lambda: object()  # type: ignore[method-assign]

    collection = store._ensure_collection()

    assert collection is fake_collection
    assert calls[0]["name"] == store.collection_name
    assert calls[0]["name"] != "rag_documents"
    assert calls[0]["metadata"]["embedding_model"] == "Qwen/Qwen3-Embedding-8B"
