"""RAG向量存储层 - 基于ChromaDB的文档块存储"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from minicrab.agent.rag.chunker import fetch_siliconflow_embeddings, is_remote_embedding_model
from minicrab.utils.helpers import ensure_dir


@dataclass
class DocumentChunk:
    """文档分块数据结构"""

    chunk_id: str
    file_path: str
    content: str
    chunk_index: int = 0
    doc_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "doc_type": self.doc_type,
            "metadata": self.metadata,
        }


@dataclass
class DocumentRecord:
    """已索引文档的元信息记录"""

    file_path: str
    file_name: str
    doc_type: str
    chunk_count: int = 0
    file_mtime: float = 0.0
    file_size: int = 0
    indexed_at: str = ""
    expires_at: str = ""
    status: str = "indexed"

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "doc_type": self.doc_type,
            "chunk_count": self.chunk_count,
            "file_mtime": self.file_mtime,
            "file_size": self.file_size,
            "indexed_at": self.indexed_at,
            "expires_at": self.expires_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DocumentRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def is_expired(self) -> bool:
        if not self.expires_at or self.status != "indexed":
            return False
        try:
            exp_time = datetime.fromisoformat(self.expires_at)
            return datetime.now(timezone.utc) > exp_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return False


class RAGStore:
    """RAG文档向量存储管理器

    负责：
    - ChromaDB集合管理（与Episode共用客户端，独立collection）
    - 文档块的CRUD操作
    - 已索引文档的元信息维护（用于去重、增量更新、过期清理）
    """

    COLLECTION_NAME = "rag_documents"
    VECTOR_STORE_REPAIR_HINT = (
        "本地 Chroma 向量库无法打开，可能是旧版本数据格式不兼容或本地数据库损坏。"
        "不要重试 force_reindex；需要先备份并重建 memory/rag/chroma 后再重新索引文档。"
    )

    def __init__(
        self,
        memory_dir: Path,
        ttl_days: int = 30,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.rag_dir = ensure_dir(memory_dir / "rag")
        self._collection = None
        self._client = None
        self.ttl_days = ttl_days
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url or "https://api.siliconflow.cn/v1"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = self._collection_name_for_model(embedding_model)
        self.registry_file = self._registry_file_for_collection(self.collection_name)
        self._registry: dict[str, DocumentRecord] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """加载文档注册表"""
        if self.registry_file.exists():
            try:
                data = json.loads(self.registry_file.read_text(encoding="utf-8"))
                self._registry = {
                    fp: DocumentRecord.from_dict(record)
                    for fp, record in data.get("documents", {}).items()
                }
            except (json.JSONDecodeError, TypeError):
                logger.warning("RAG document registry corrupted, starting fresh")
                self._registry = {}

    def _save_registry(self) -> None:
        """保存文档注册表"""
        data = {
            "documents": {
                fp: record.to_dict() for fp, record in self._registry.items()
            },
            "total_documents": len(self._registry),
            "total_chunks": sum(r.chunk_count for r in self._registry.values()),
        }
        self.registry_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _ensure_collection(self) -> Any:
        """延迟初始化ChromaDB集合"""
        if self._collection is not None:
            return self._collection

        try:
            self._client = self._create_chroma_client()

            embedding_fn = self._create_embedding_function()
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
            )
            logger.info(f"RAG vector store ready: {self.rag_dir / 'chroma'} (model={self.embedding_model})")
            return self._collection

        except ImportError:
            logger.warning(
                "ChromaDB not installed. Install from source: pip install -e '.[vector]'"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize RAG vector store: {e}")
            raise self._vector_store_error(e) from e

    def _create_chroma_client(self) -> Any:
        """创建 Chroma 客户端，并关闭匿名遥测噪声。"""
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        import chromadb

        chroma_dir = self.rag_dir / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        try:
            from chromadb.config import Settings
        except (ImportError, ModuleNotFoundError):
            return chromadb.PersistentClient(path=str(chroma_dir))
        return chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(
                anonymized_telemetry=False,
                chroma_product_telemetry_impl=(
                    "minicrab.agent.rag.telemetry.NoOpProductTelemetryClient"
                ),
                chroma_telemetry_impl=(
                    "minicrab.agent.rag.telemetry.NoOpProductTelemetryClient"
                ),
            ),
        )

    @classmethod
    def _vector_store_error(cls, error: BaseException) -> RuntimeError:
        """把底层 Chroma 异常转换成模型能理解的错误。"""
        if isinstance(error, KeyError) and error.args and error.args[0] == "_type":
            return RuntimeError(cls.VECTOR_STORE_REPAIR_HINT)
        return RuntimeError(f"RAG vector store unavailable: {error}")

    def _create_embedding_function(self) -> Any:
        """创建嵌入函数，支持自定义模型和远程API"""
        import chromadb

        model = self.embedding_model

        if model == "all-MiniLM-L6-v2":
            return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()

        if is_remote_embedding_model(model):
            return self._create_siliconflow_embedding_function()

        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(model, cache_folder=None)

            class CustomEmbeddingFunction(chromadb.EmbeddingFunction):
                def __call__(self, input: list[str]) -> list[list[float]]:
                    embeddings = st_model.encode(
                        input, normalize_embeddings=True, show_progress_bar=False
                    )
                    return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

            logger.info(f"Using custom embedding model: {model}")
            return CustomEmbeddingFunction()

        except ImportError:
            logger.warning(
                f"sentence-transformers not available, falling back to default for model '{model}'"
            )
            return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            logger.warning(f"Failed to load embedding model '{model}': {e}, using default")
            return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()

    @classmethod
    def _collection_name_for_model(cls, embedding_model: str) -> str:
        """根据 embedding 模型选择隔离的 collection 名称。"""
        if embedding_model == "all-MiniLM-L6-v2":
            return cls.COLLECTION_NAME

        clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", embedding_model.lower()).strip("_")
        clean = clean[:24] or "custom"
        digest = hashlib.sha1(embedding_model.encode("utf-8")).hexdigest()[:8]
        return f"{cls.COLLECTION_NAME}_{clean}_{digest}"[:63]

    def _registry_file_for_collection(self, collection_name: str) -> Path:
        """返回当前 collection 对应的文档注册表路径。"""
        if collection_name == self.COLLECTION_NAME:
            return self.rag_dir / "document_registry.json"
        return self.rag_dir / f"{collection_name}_registry.json"

    def _all_registry_files(self) -> list[Path]:
        """返回所有 RAG 注册表文件。"""
        files: list[Path] = []
        legacy = self.rag_dir / "document_registry.json"
        if legacy.exists():
            files.append(legacy)
        for path in self.rag_dir.glob(f"{self.COLLECTION_NAME}*_registry.json"):
            if path not in files:
                files.append(path)
        if self.registry_file not in files:
            files.append(self.registry_file)
        return files

    def _read_registry_totals(self) -> tuple[int, int]:
        """统计所有注册表里的文档数和分块数。"""
        total_docs = 0
        total_chunks = 0
        for path in self._all_registry_files():
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning(f"Failed to read RAG registry while clearing: {path}")
                continue
            documents = data.get("documents", {})
            if not isinstance(documents, dict):
                continue
            total_docs += len(documents)
            for record in documents.values():
                if isinstance(record, dict):
                    total_chunks += int(record.get("chunk_count", 0) or 0)
        return total_docs, total_chunks

    def _clear_all_registries(self) -> None:
        """清空所有 RAG 注册表文件。"""
        empty = json.dumps(
            {"documents": {}, "total_documents": 0, "total_chunks": 0},
            ensure_ascii=False,
            indent=2,
        )
        for path in self._all_registry_files():
            path.write_text(empty, encoding="utf-8")
        self._registry.clear()

    def _create_siliconflow_embedding_function(self) -> Any:
        """创建SiliconFlow API嵌入函数"""
        import chromadb
        import httpx

        model = self.embedding_model.replace("siliconflow:", "")
        api_key = self.embedding_api_key
        base_url = self.embedding_base_url

        if not api_key:
            logger.warning("SiliconFlow API key not provided, falling back to default embedding")
            return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()

        class SiliconFlowEmbeddingFunction(chromadb.EmbeddingFunction):
            def __call__(self, input: list[str]) -> list[list[float]]:
                try:
                    return fetch_siliconflow_embeddings(
                        texts=input,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                    )
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text[:500] if exc.response is not None else str(exc)
                    status = exc.response.status_code if exc.response is not None else "unknown"
                    raise RuntimeError(
                        f"SiliconFlow embedding request failed: HTTP {status}: {detail}"
                    ) from exc
                except httpx.HTTPError as exc:
                    raise RuntimeError(f"SiliconFlow embedding request failed: {exc}") from exc
                except ValueError as exc:
                    raise RuntimeError(f"SiliconFlow embedding returned invalid JSON: {exc}") from exc

            def __str__(self):
                return f"SiliconFlowEmbeddingFunction(model={model})"

        logger.info(f"Using SiliconFlow embedding model: {model}")
        return SiliconFlowEmbeddingFunction()

    @staticmethod
    def _make_chunk_id(file_path: str, chunk_index: int) -> str:
        """生成唯一的chunk ID"""
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
        return f"{path_hash}_{chunk_index:04d}"

    def get_document_record(self, file_path: str) -> DocumentRecord | None:
        """获取文件的索引记录"""
        return self._registry.get(file_path)

    def is_document_indexed(self, file_path: str, current_mtime: float | None = None) -> bool:
        """检查文件是否已索引（可选校验修改时间和过期状态）"""
        record = self._registry.get(file_path)
        if record is None:
            return False
        if record.is_expired:
            return False
        if current_mtime is not None and record.file_mtime != current_mtime:
            return False
        return record.status == "indexed"

    def register_document(
        self,
        file_path: str,
        doc_type: str = "text",
        chunk_count: int = 0,
        file_mtime: float = 0.0,
        file_size: int = 0,
    ) -> DocumentRecord:
        """注册/更新文档记录，自动计算过期时间"""
        now = datetime.now(timezone.utc)
        expires_at = ""

        if self.ttl_days > 0:
            from datetime import timedelta
            expires_at = (now + timedelta(days=self.ttl_days)).isoformat()

        path_obj = Path(file_path)
        record = DocumentRecord(
            file_path=file_path,
            file_name=path_obj.name,
            doc_type=doc_type,
            chunk_count=chunk_count,
            file_mtime=file_mtime,
            file_size=file_size,
            indexed_at=now.isoformat(),
            expires_at=expires_at,
            status="indexed",
        )
        self._registry[file_path] = record
        self._save_registry()
        return record

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """批量添加文档块到向量存储

        Returns:
            成功添加的块数量

        Raises:
            RuntimeError: 当upsert完全失败时（非部分失败）
        """
        if not chunks:
            return 0

        collection = self._ensure_collection()

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            metadatas.append({
                "file_path": chunk.file_path,
                "file_name": Path(chunk.file_path).name,
                "chunk_index": chunk.chunk_index,
                "doc_type": chunk.doc_type,
                **chunk.metadata,
            })

        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            logger.debug(f"Added {len(chunks)} chunks to RAG store")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to upsert {len(chunks)} chunks: {e}")
            raise RuntimeError(f"Vector store upsert failed: {e}") from e

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        exclude_expired: bool = True,
    ) -> list[dict[str, Any]]:
        """向量相似度搜索

        Args:
            query: 查询文本
            n_results: 返回结果数上限
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            exclude_expired: 是否排除已过期文档的结果

        Returns:
            搜索结果列表，每项包含 id, document, metadata, distance
        """
        collection = self._ensure_collection()

        try:
            kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": n_results,
            }
            if where:
                kwargs["where"] = where
            if where_document:
                kwargs["where_document"] = where_document

            results = collection.query(**kwargs)

            if not results or not results.get("ids"):
                return []

            items = []
            ids_list = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, chunk_id in enumerate(ids_list):
                meta = metadatas[i] if i < len(metadatas) else {}
                file_path = meta.get("file_path", "")

                if exclude_expired and file_path:
                    record = self._registry.get(file_path)
                    if record and record.is_expired:
                        continue

                items.append({
                    "id": chunk_id,
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": meta,
                    "distance": distances[i] if i < len(distances) else 1.0,
                })

            return items

        except Exception as e:
            logger.warning(f"RAG vector search failed: {e}")
            return []

    def get_by_file(self, file_path: str) -> list[dict[str, Any]]:
        """获取指定文件的所有文档块"""
        collection = self._ensure_collection()

        try:
            results = collection.get(
                where={"file_path": file_path},
                include=["documents", "metadatas"],
            )

            if not results or not results.get("ids"):
                return []

            items = []
            for i, chunk_id in enumerate(results["ids"]):
                items.append({
                    "id": chunk_id,
                    "document": results["documents"][i] if i < len(results.get("documents", [])) else "",
                    "metadata": results["metadatas"][i] if i < len(results.get("metadatas", [])) else {},
                })

            return sorted(items, key=lambda x: x["metadata"].get("chunk_index", 0))

        except Exception as e:
            logger.error(f"Failed to get chunks for {file_path}: {e}")
            return []

    def delete_by_file(self, file_path: str) -> int:
        """删除指定文件的所有文档块

        Returns:
            删除的块数量
        """
        collection = self._ensure_collection()

        try:
            chunks = self.get_by_file(file_path)
            if not chunks:
                return 0

            chunk_ids = [c["id"] for c in chunks]
            collection.delete(ids=chunk_ids)

            if file_path in self._registry:
                del self._registry[file_path]
                self._save_registry()

            logger.debug(f"Deleted {len(chunk_ids)} chunks for {file_path}")
            return len(chunk_ids)

        except Exception as e:
            logger.error(f"Failed to delete chunks for {file_path}: {e}")
            return 0

    def count(self) -> int:
        """返回总文档块数量"""
        collection = self._ensure_collection()
        return collection.count()

    def cleanup_expired(self) -> dict[str, Any]:
        """清理所有已过期文档（删除向量数据 + 注册记录）

        Returns:
            清理统计信息
        """
        expired_files = [
            (fp, rec) for fp, rec in self._registry.items()
            if rec.is_expired
        ]

        if not expired_files:
            return {"cleaned": 0, "files": []}

        removed_chunks = 0
        cleaned_files = []

        for file_path, record in expired_files:
            deleted = self.delete_by_file(file_path)
            removed_chunks += deleted
            cleaned_files.append({
                "file_name": record.file_name,
                "chunks_removed": deleted,
                "expired_at": record.expires_at,
            })

        logger.info(
            f"RAG cleanup: {len(cleaned_files)} expired documents removed, "
            f"{removed_chunks} chunks freed"
        )
        return {"cleaned": len(cleaned_files), "chunks_freed": removed_chunks, "files": cleaned_files}

    def get_stats(self) -> dict[str, Any]:
        """获取RAG存储统计信息"""
        total_docs = len(self._registry)
        expired_count = sum(1 for r in self._registry.values() if r.is_expired)
        active_count = total_docs - expired_count
        total_chunks = sum(r.chunk_count for r in self._registry.values())

        vector_store_error = None
        vector_store_status = "ready"
        try:
            vector_count = self.count()
        except Exception as exc:
            vector_count = -1
            vector_store_status = "unavailable"
            vector_store_error = str(exc)

        return {
            "total_documents": total_docs,
            "active_documents": active_count,
            "expired_documents": expired_count,
            "total_chunks": total_chunks,
            "vector_count": vector_count,
            "vector_store_status": vector_store_status,
            "vector_store_error": vector_store_error,
            "repair_required": vector_store_status != "ready",
            "ttl_days": self.ttl_days,
            "embedding_model": self.embedding_model,
            "collection_name": self.collection_name,
            "documents": [
                {
                    "file_name": r.file_name,
                    "doc_type": r.doc_type,
                    "chunk_count": r.chunk_count,
                    "indexed_at": r.indexed_at,
                    "expires_at": r.expires_at,
                    "status": "expired" if r.is_expired else r.status,
                }
                for r in self._registry.values()
            ],
        }

    def clear_all(self) -> dict[str, Any]:
        """清空所有RAG数据（所有 RAG collection + 注册表）"""
        doc_count, total_chunks = self._read_registry_totals()
        collections_removed = 0
        vectors_removed = 0
        try:
            self._client = self._create_chroma_client()
            for collection in self._client.list_collections():
                name = collection if isinstance(collection, str) else collection.name
                if not name.startswith(self.COLLECTION_NAME):
                    continue
                try:
                    current = self._client.get_collection(name=name)
                    vectors_removed += int(current.count() or 0)
                except Exception as e:
                    logger.warning(f"Failed to count RAG collection before clearing {name}: {e}")
                self._client.delete_collection(name=name)
                collections_removed += 1
        except Exception as e:
            logger.warning(f"Failed to clear vector store: {e}")

        self._clear_all_registries()
        self._collection = None

        logger.info(
            f"RAG store cleared: {doc_count} documents, {total_chunks} chunks removed, "
            f"{collections_removed} collections removed"
        )
        return {
            "documents_removed": doc_count,
            "chunks_removed": total_chunks,
            "vectors_removed": vectors_removed,
            "collections_removed": collections_removed,
        }
