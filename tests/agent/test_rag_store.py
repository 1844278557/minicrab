"""测试 RAG 存储的模型隔离行为。"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from nanobot.agent.rag.store import RAGStore
from nanobot.agent.tools.rag import IndexDocumentTool


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


def test_ensure_collection_reports_legacy_chroma_config_error(monkeypatch, tmp_path) -> None:
    """旧 Chroma 配置缺少 _type 时，返回可理解的修复提示。"""
    class FakeClient:
        def __init__(self, path: str):
            self.path = path

        def get_or_create_collection(self, **_kwargs):
            raise KeyError("_type")

    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=FakeClient))
    store = RAGStore(tmp_path / "memory", embedding_model="Qwen/Qwen3-Embedding-8B")
    store._create_embedding_function = lambda: object()  # type: ignore[method-assign]

    try:
        store._ensure_collection()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert "Chroma 向量库无法打开" in message
    assert "不要重试 force_reindex" in message


def test_stats_marks_vector_store_unavailable(tmp_path) -> None:
    """列表统计不能把向量库失败伪装成正常空库。"""
    store = RAGStore(tmp_path / "memory", embedding_model="Qwen/Qwen3-Embedding-8B")
    store.count = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]

    stats = store.get_stats()

    assert stats["vector_count"] == -1
    assert stats["vector_store_status"] == "unavailable"
    assert stats["vector_store_error"] == "boom"
    assert stats["repair_required"] is True


def test_index_tool_returns_non_retryable_rag_error(tmp_path) -> None:
    """索引失败时明确告诉模型不能靠 force_reindex 解决。"""
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello", encoding="utf-8")
    store = RAGStore(tmp_path / "memory", embedding_model="Qwen/Qwen3-Embedding-8B")
    tool = IndexDocumentTool(store)
    tool.indexer.index_file = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError(RAGStore.VECTOR_STORE_REPAIR_HINT)
    )

    import asyncio

    payload = json.loads(asyncio.run(tool.execute(file_path=str(file_path))))

    assert payload["code"] == "rag_vector_store_unavailable"
    assert payload["retryable"] is False
    assert payload["force_reindex_will_help"] is False


def test_clear_all_removes_every_rag_collection_and_registry(monkeypatch, tmp_path) -> None:
    """清空知识库时要覆盖所有 RAG collection，而不是只清当前模型。"""
    store = RAGStore(tmp_path / "memory", embedding_model="Qwen/Qwen3-Embedding-8B")
    legacy_registry = store.rag_dir / "document_registry.json"
    current_registry = store.registry_file
    legacy_registry.write_text(
        json.dumps({
            "documents": {"old.txt": {"chunk_count": 2}},
            "total_documents": 1,
            "total_chunks": 2,
        }),
        encoding="utf-8",
    )
    current_registry.write_text(
        json.dumps({
            "documents": {"new.txt": {"chunk_count": 3}},
            "total_documents": 1,
            "total_chunks": 3,
        }),
        encoding="utf-8",
    )

    deleted: list[str] = []

    class FakeCollection:
        def __init__(self, count: int):
            self._count = count

        def count(self) -> int:
            return self._count

    class FakeClient:
        def __init__(self, path: str):
            self.path = path

        def list_collections(self):
            return [
                SimpleNamespace(name="rag_documents"),
                store.collection_name,
                SimpleNamespace(name="episodes"),
            ]

        def get_collection(self, name: str):
            return FakeCollection({"rag_documents": 2, store.collection_name: 3}[name])

        def delete_collection(self, name: str) -> None:
            deleted.append(name)

    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=FakeClient))

    result = store.clear_all()

    assert deleted == ["rag_documents", store.collection_name]
    assert result["documents_removed"] == 2
    assert result["chunks_removed"] == 5
    assert result["vectors_removed"] == 5
    assert result["collections_removed"] == 2
    assert json.loads(legacy_registry.read_text(encoding="utf-8"))["documents"] == {}
    assert json.loads(current_registry.read_text(encoding="utf-8"))["documents"] == {}
