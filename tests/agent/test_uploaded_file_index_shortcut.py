"""测试上传文件索引确认的系统级承接。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from minicrab.agent.loop import AgentLoop
from minicrab.bus.events import InboundMessage
from minicrab.bus.queue import MessageBus
from minicrab.providers.base import GenerationSettings
from minicrab.session.manager import Session


def _make_loop(tmp_path: Path) -> AgentLoop:
    """创建测试用 AgentLoop。"""
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.estimate_prompt_tokens.return_value = (0, "test-counter")
    provider.chat_with_retry = AsyncMock()
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop._schedule_post_turn_maintenance = MagicMock()
    loop.tools.execute = AsyncMock(return_value=json.dumps({
        "success": True,
        "chunk_count": 3,
    }))
    return loop


def _uploaded_file_message(upload: Path) -> str:
    """构造渠道保存到会话里的文件消息。"""
    return f"[file: {upload.name}]\n[File: source: {upload}]"


def _assistant_index_question() -> dict:
    """构造上一轮助手通过 message 工具发出的索引确认话术。"""
    return {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "message",
            "arguments": json.dumps({
                "content": "请确认是否索引到 RAG 知识库，回复“是”或“确认”。",
            }, ensure_ascii=False),
        },
    }


@pytest.mark.asyncio
async def test_short_yes_indexes_recent_uploaded_file_without_llm(tmp_path: Path) -> None:
    """上一轮询问索引后，用户回复“是”会直接索引最近上传文件。"""
    upload = tmp_path / "信息管理研究方法课程报告.pdf"
    upload.write_text("fake pdf", encoding="utf-8")
    loop = _make_loop(tmp_path)
    session = loop.sessions.get_or_create("weixin:chat")
    session.add_message("user", _uploaded_file_message(upload))
    session.add_message("assistant", "", tool_calls=[_assistant_index_question()])

    response = await loop._process_message(InboundMessage(
        channel="weixin",
        sender_id="user",
        chat_id="chat",
        content="是",
    ))

    assert response is not None
    assert f"已把 {upload.name} 索引到知识库" in response.content
    loop.tools.execute.assert_awaited_once_with("index_document", {"file_path": str(upload)})
    loop.provider.chat_with_retry.assert_not_awaited()
    assert session.messages[-2]["content"] == "是"
    assert upload.name in session.messages[-1]["content"]


@pytest.mark.asyncio
async def test_direct_index_latest_upload_request_indexes_file(tmp_path: Path) -> None:
    """用户直接说索引刚上传的文件时，也会直接执行索引。"""
    upload = tmp_path / "课程报告.docx"
    upload.write_text("fake docx", encoding="utf-8")
    loop = _make_loop(tmp_path)
    session = loop.sessions.get_or_create("weixin:chat")
    session.add_message("user", _uploaded_file_message(upload))

    response = await loop._process_message(InboundMessage(
        channel="weixin",
        sender_id="user",
        chat_id="chat",
        content="索引刚才上传的文件",
    ))

    assert response is not None
    assert f"已把 {upload.name} 索引到知识库" in response.content
    loop.tools.execute.assert_awaited_once_with("index_document", {"file_path": str(upload)})
    loop.provider.chat_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_index_request_ignores_plain_internal_paths(tmp_path: Path) -> None:
    """普通历史路径不是上传文件，不能被“刚才上传”误选中。"""
    internal = tmp_path / "USER.md"
    upload = tmp_path / "课程报告.docx"
    internal.write_text("user memory", encoding="utf-8")
    upload.write_text("fake docx", encoding="utf-8")
    loop = _make_loop(tmp_path)
    session = loop.sessions.get_or_create("weixin:chat")
    session.add_message("assistant", f"读取过内部文件：{internal}")
    session.add_message("user", _uploaded_file_message(upload))

    response = await loop._process_message(InboundMessage(
        channel="weixin",
        sender_id="user",
        chat_id="chat",
        content="索引刚才上传的文件",
    ))

    assert response is not None
    assert f"已把 {upload.name} 索引到知识库" in response.content
    loop.tools.execute.assert_awaited_once_with("index_document", {"file_path": str(upload)})


@pytest.mark.asyncio
async def test_short_yes_without_recent_index_question_is_not_intercepted(tmp_path: Path) -> None:
    """没有最近索引确认话术时，单独“是”不会被误拦截。"""
    upload = tmp_path / "课程报告.docx"
    upload.write_text("fake docx", encoding="utf-8")
    loop = _make_loop(tmp_path)
    session = Session(key="weixin:chat")
    session.add_message("user", _uploaded_file_message(upload))
    msg = InboundMessage(channel="weixin", sender_id="user", chat_id="chat", content="是")

    response = await loop._handle_uploaded_file_index_shortcut(msg, session, "是")

    assert response is None
    loop.tools.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_shortcut_respects_index_document_approval_rule(tmp_path: Path) -> None:
    """用户配置索引需要确认时，快捷分支不会直接执行工具。"""
    (tmp_path / "permissions.json").write_text(
        json.dumps({"allow": [], "ask": ["index_document(*)"], "deny": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    upload = tmp_path / "课程报告.docx"
    upload.write_text("fake docx", encoding="utf-8")
    loop = _make_loop(tmp_path)
    session = loop.sessions.get_or_create("weixin:chat")
    session.add_message("user", _uploaded_file_message(upload))

    response = await loop._process_message(InboundMessage(
        channel="weixin",
        sender_id="user",
        chat_id="chat",
        content="索引刚才上传的文件",
    ))

    assert response is not None
    assert "要我索引这个文档吗？" in response.content
    loop.tools.execute.assert_not_awaited()
