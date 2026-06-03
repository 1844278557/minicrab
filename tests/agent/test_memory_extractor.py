"""测试回复后自动提取长期工作记忆。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.memory_extractor import MemoryExtractor
from nanobot.providers.base import LLMResponse
from nanobot.session.manager import Session, SessionManager


def _session_with_messages() -> Session:
    """创建包含一轮对话的测试 session。"""
    session = Session(key="web:default")
    session.add_message("user", "以后 Web 端审批都用弹窗，不进入普通历史消息。")
    session.add_message("assistant", "好的，我会按这个规则处理。")
    return session


@pytest.mark.asyncio
async def test_memory_extractor_updates_memory_md_and_cursor(tmp_path: Path) -> None:
    """合法 JSON 更新只写 MEMORY.md，并推进游标。"""
    session = _session_with_messages()
    sessions = SessionManager(tmp_path)
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "should_update": True,
            "memory_update": "# Working Memory\n\n## Project Decisions\n\n- Web approval uses modal and stays out of normal history.",
            "reason": "Captured durable Web approval decision.",
        })
    ))
    extractor = MemoryExtractor(tmp_path, provider, "test-model")

    updated = await extractor.extract_new(session, sessions)

    assert updated is True
    assert "Web approval uses modal" in (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert not (tmp_path / "USER.md").exists()
    assert session.metadata["memory_extracted_until"] == len(session.messages)


@pytest.mark.asyncio
async def test_memory_extractor_skips_invalid_json_and_advances_cursor(tmp_path: Path) -> None:
    """LLM 没返回严格 JSON 时不写记忆，但不会反复卡在同一段消息。"""
    session = _session_with_messages()
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="这里不是 JSON"))
    extractor = MemoryExtractor(tmp_path, provider, "test-model")

    updated = await extractor.extract_new(session)

    assert updated is False
    assert not (tmp_path / "memory" / "MEMORY.md").exists()
    assert session.metadata["memory_extracted_until"] == len(session.messages)


@pytest.mark.asyncio
async def test_memory_extractor_no_update_does_not_create_memory_file(tmp_path: Path) -> None:
    """模型判断没有长期价值时不创建 MEMORY.md。"""
    session = _session_with_messages()
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "should_update": False,
            "memory_update": "",
            "reason": "No durable memory.",
        })
    ))
    extractor = MemoryExtractor(tmp_path, provider, "test-model")

    updated = await extractor.extract_new(session)

    assert updated is False
    assert not (tmp_path / "memory" / "MEMORY.md").exists()
    assert session.metadata["memory_extracted_until"] == len(session.messages)


@pytest.mark.asyncio
async def test_memory_extractor_start_index_skips_old_history_without_cursor(tmp_path: Path) -> None:
    """没有旧游标时，调用方可以指定只扫描当前轮新增消息。"""
    session = Session(key="web:default")
    session.add_message("user", "旧消息不应该被首次升级扫描。")
    session.add_message("assistant", "旧回复。")
    start_index = len(session.messages)
    session.add_message("user", "新增消息：Web 审批弹窗是长期规则。")
    session.add_message("assistant", "收到。")

    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "should_update": False,
            "memory_update": "",
            "reason": "No update in test.",
        })
    ))
    extractor = MemoryExtractor(tmp_path, provider, "test-model")

    await extractor.extract_new(session, start_index=start_index)

    _, kwargs = provider.chat_with_retry.await_args
    prompt_text = kwargs["messages"][1]["content"]
    assert "新增消息" in prompt_text
    assert "旧消息不应该" not in prompt_text
