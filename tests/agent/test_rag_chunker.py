"""测试 RAG 分块器选择和远程 embedding 参数传递。"""

from __future__ import annotations

from minicrab.agent.rag.chunker import (
    MarkdownHeaderSplitter,
    SemanticSplitter,
    get_chunker_for_file,
)


def test_markdown_chunker_ignores_embedding_kwargs() -> None:
    """Markdown 分块器不能收到 embedding 专用参数。"""
    chunker = get_chunker_for_file(
        "note.md",
        embedding_model="BAAI/bge-large-zh-v1.5",
        embedding_api_key="test-key",
        embedding_base_url="https://api.siliconflow.cn/v1",
    )

    chunks = chunker.chunk("# 标题\n\n这里是正文。", file_path="note.md")

    assert isinstance(chunker, MarkdownHeaderSplitter)
    assert len(chunks) == 1
    assert "这里是正文" in chunks[0].content


def test_baai_semantic_chunker_uses_siliconflow_embeddings(monkeypatch) -> None:
    """BAAI 模型用于语义分块时应走远程 API 分支，而不是本地 HF 模型。"""
    calls: list[list[str]] = []

    def fake_embeddings(self, sentences: list[str]) -> list[list[float]]:
        calls.append(sentences)
        return [[1.0, 0.0], [1.0, 0.0]]

    monkeypatch.setattr(SemanticSplitter, "_get_siliconflow_embeddings", fake_embeddings)
    chunker = get_chunker_for_file(
        "report.pdf",
        embedding_model="BAAI/bge-large-zh-v1.5",
        embedding_api_key="test-key",
        embedding_base_url="https://api.siliconflow.cn/v1",
    )

    chunks = chunker.chunk("第一句。 第二句。", file_path="report.pdf")

    assert isinstance(chunker, SemanticSplitter)
    assert calls == [["第一句。", "第二句。"]]
    assert len(chunks) == 1
