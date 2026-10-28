"""测试回复后受限 MemoryAgent 自动维护长期记忆。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minicrab.agent.loop import AgentLoop
from minicrab.agent.memory_extractor import MemoryAgent
from minicrab.bus.queue import MessageBus
from minicrab.providers.base import GenerationSettings, LLMResponse
from minicrab.session.manager import Session, SessionManager


def _session_with_messages() -> Session:
    """创建包含一轮对话的测试 session。"""
    session = Session(key="web:default")
    session.add_message("user", "以后 Web 端审批都用弹窗，不进入普通历史消息。")
    session.add_message("assistant", "好的，我会按这个规则处理。")
    return session


def _loop_provider() -> MagicMock:
    """创建 AgentLoop 测试所需的 provider。"""
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.estimate_prompt_tokens.return_value = (0, "test-counter")
    return provider


def test_agent_loop_uses_dedicated_memory_provider(tmp_path: Path) -> None:
    """AgentLoop 可以让记忆提取使用独立模型。"""
    provider = _loop_provider()
    memory_provider = _loop_provider()

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="main-model",
        memory_provider=memory_provider,
        memory_model="memory-model",
    )

    assert loop.memory_agent.provider is memory_provider
    assert loop.memory_agent.model == "memory-model"


@pytest.mark.asyncio
async def test_memory_agent_updates_memory_md_and_cursor(tmp_path: Path) -> None:
    """合法 JSON 可以更新 MEMORY.md，并推进游标。"""
    session = _session_with_messages()
    sessions = SessionManager(tmp_path)
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "should_update": True,
            "operations": [{
                "target": "memory",
                "op": "append",
                "section": "Project Decisions",
                "match": "",
                "content": "- [decision][2026-06-04] Web approval uses modal and stays out of normal history.",
                "confidence": "high",
                "reason": "Captured durable Web approval decision.",
            }],
            "reason": "Captured durable Web approval decision.",
        })
    ))
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    updated = await extractor.extract_new(session, sessions)

    assert updated is True
    memory_text = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "- [decision][2026-06-04] Web approval uses modal" in memory_text
    assert not (tmp_path / "USER.md").exists()
    assert session.metadata["memory_extracted_until"] == len(session.messages)
    _, kwargs = provider.chat_with_retry.await_args
    assert kwargs["response_format"] == {"type": "json_object"}


def test_memory_agent_prompt_describes_memory_entry_format(tmp_path: Path) -> None:
    """MemoryAgent 提示词要求 MEMORY.md 使用轻量类型和日期前缀。"""
    provider = AsyncMock()
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    messages = extractor._build_prompt("USER: Web 审批用弹窗。")
    prompt = messages[1]["content"]
    system_prompt = messages[0]["content"]

    assert f"当前日期是 {datetime.now().date().isoformat()}" in prompt
    assert '"operations"' in prompt
    assert "只输出 append、replace、delete" in prompt
    assert "只有 confidence=high 的操作才会执行" in prompt
    assert "自动写 USER.md 必须非常保守" in system_prompt
    assert "- [rule][YYYY-MM-DD] 内容" in prompt
    assert "- [decision][YYYY-MM-DD] 内容" in prompt
    assert "- [lesson][YYYY-MM-DD] 内容" in prompt
    assert "- [info][YYYY-MM-DD] 内容" in prompt
    assert "不新增重复条目" in prompt


@pytest.mark.asyncio
async def test_memory_agent_updates_user_md(tmp_path: Path) -> None:
    """自动提取可以把稳定用户信息写入 USER.md。"""
    session = Session(key="web:default")
    session.add_message("user", "以后都用简洁中文回复我。")
    session.add_message("assistant", "好的。")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "should_update": True,
            "operations": [{
                "target": "user",
                "op": "append",
                "section": "Preferences",
                "match": "",
                "content": "- Prefer concise Chinese replies.",
                "confidence": "high",
                "reason": "Captured stable communication preference.",
            }],
            "reason": "User preference is durable.",
        })
    ))
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    updated = await extractor.extract_new(session)

    assert updated is True
    assert "concise Chinese" in (tmp_path / "USER.md").read_text(encoding="utf-8")
    assert not (tmp_path / "memory" / "MEMORY.md").exists()


@pytest.mark.asyncio
async def test_memory_agent_updates_user_and_memory_md(tmp_path: Path) -> None:
    """一次提取可以同时更新 USER.md 和 MEMORY.md。"""
    session = _session_with_messages()
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "should_update": True,
            "operations": [
                {
                    "target": "user",
                    "op": "append",
                    "section": "Preferences",
                    "match": "",
                    "content": "- Prefer concise replies.",
                    "confidence": "high",
                    "reason": "Captured user preference.",
                },
                {
                    "target": "memory",
                    "op": "append",
                    "section": "Project Decisions",
                    "match": "",
                    "content": "- [decision][2026-06-04] Web approval uses modal.",
                    "confidence": "high",
                    "reason": "Captured project decision.",
                },
            ],
            "reason": "Both memories changed.",
        })
    ))
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    updated = await extractor.extract_new(session)

    assert updated is True
    assert "Prefer concise replies" in (tmp_path / "USER.md").read_text(encoding="utf-8")
    assert "Web approval uses modal" in (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_memory_agent_skips_invalid_json_and_advances_cursor(tmp_path: Path) -> None:
    """LLM 没返回严格 JSON 时不写记忆，但不会反复卡在同一段消息。"""
    session = _session_with_messages()
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="这里不是 JSON"))
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    with patch("minicrab.agent.memory_extractor.logger.warning") as warning_log:
        updated = await extractor.extract_new(session)

    assert updated is False
    assert not (tmp_path / "memory" / "MEMORY.md").exists()
    assert session.metadata["memory_extracted_until"] == len(session.messages)
    assert any("plain-text non-JSON response" in str(call) for call in warning_log.call_args_list)


@pytest.mark.asyncio
async def test_memory_agent_rejects_json_code_block(tmp_path: Path) -> None:
    """模型把 JSON 包在代码块里时，按非严格 JSON 拒绝。"""
    session = Session(key="web:default")
    session.add_message("user", "以后都用简洁中文回复我。")
    session.add_message("assistant", "好的。")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="```json\n"
        + json.dumps({
            "should_update": True,
            "operations": [{
                "target": "user",
                "op": "append",
                "section": "Preferences",
                "match": "",
                "content": "- Prefer concise Chinese replies.",
                "confidence": "high",
                "reason": "Captured stable preference.",
            }],
            "reason": "Stable preference.",
        })
        + "\n```"
    ))
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    with patch("minicrab.agent.memory_extractor.logger.warning") as warning_log:
        updated = await extractor.extract_new(session)

    assert updated is False
    assert not (tmp_path / "USER.md").exists()
    assert any("markdown code block" in str(call) for call in warning_log.call_args_list)


@pytest.mark.asyncio
async def test_memory_agent_rejects_json_with_surrounding_text(tmp_path: Path) -> None:
    """模型在 JSON 前后加说明文字时，按非严格 JSON 拒绝。"""
    session = _session_with_messages()
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=(
            "判断结果如下：\n"
            + json.dumps({
                "should_update": False,
                "operations": [],
                "reason": "No durable memory.",
            })
            + "\n以上。"
        )
    ))
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    with patch("minicrab.agent.memory_extractor.logger.warning") as warning_log:
        updated = await extractor.extract_new(session)

    assert updated is False
    assert not (tmp_path / "USER.md").exists()
    assert any("JSON-like response" in str(call) for call in warning_log.call_args_list)


@pytest.mark.asyncio
async def test_memory_agent_rejects_plain_text_no_update(tmp_path: Path) -> None:
    """模型用自然语言表达无长期记忆时，仍按格式违规处理。"""
    session = _session_with_messages()
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="不需要更新长期记忆。"))
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    with patch("minicrab.agent.memory_extractor.logger.warning") as warning_log:
        with patch("minicrab.agent.memory_extractor.logger.info") as info_log:
            updated = await extractor.extract_new(session)

    assert updated is False
    assert any("plain-text non-JSON response" in str(call) for call in warning_log.call_args_list)
    assert not any("plain-text no-update explanation" in str(call) for call in info_log.call_args_list)


@pytest.mark.asyncio
async def test_memory_agent_no_update_does_not_create_memory_files(tmp_path: Path) -> None:
    """模型判断没有长期价值时不创建记忆文件。"""
    session = _session_with_messages()
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "should_update": False,
            "operations": [],
            "reason": "No durable memory.",
        })
    ))
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    with patch("minicrab.agent.memory_extractor.logger.info") as info_log:
        updated = await extractor.extract_new(session)

    assert updated is False
    assert not (tmp_path / "memory" / "MEMORY.md").exists()
    assert not (tmp_path / "USER.md").exists()
    assert session.metadata["memory_extracted_until"] == len(session.messages)
    assert any("no durable memory" in str(call) for call in info_log.call_args_list)


@pytest.mark.asyncio
async def test_memory_agent_start_index_skips_old_history_without_cursor(tmp_path: Path) -> None:
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
            "operations": [],
            "reason": "No update in test.",
        })
    ))
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    await extractor.extract_new(session, start_index=start_index)

    _, kwargs = provider.chat_with_retry.await_args
    prompt_text = kwargs["messages"][1]["content"]
    assert "新增消息" in prompt_text
    assert "旧消息不应该" not in prompt_text


@pytest.mark.asyncio
async def test_memory_agent_replaces_conflicting_memory(tmp_path: Path) -> None:
    """明确纠正旧记忆时，MemoryAgent 只替换精确匹配的旧 bullet。"""
    memory_file = tmp_path / "memory" / "MEMORY.md"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(
        "# Working Memory\n\n## Project Decisions\n\n"
        "- [decision][2026-06-01] USER.md writes require approval.\n",
        encoding="utf-8",
    )
    session = _session_with_messages()
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "should_update": True,
            "operations": [{
                "target": "memory",
                "op": "replace",
                "section": "Project Decisions",
                "match": "- [decision][2026-06-01] USER.md writes require approval.",
                "content": "- [decision][2026-06-04] USER.md writes do not require approval.",
                "confidence": "high",
                "reason": "New project decision replaces the old one.",
            }],
            "reason": "Conflict resolved.",
        })
    ))
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    updated = await extractor.extract_new(session)

    text = memory_file.read_text(encoding="utf-8")
    assert updated is True
    assert "do not require approval" in text
    assert "- [decision][2026-06-01] USER.md writes require approval." not in text


@pytest.mark.asyncio
async def test_memory_agent_deletes_explicitly_forgotten_memory(tmp_path: Path) -> None:
    """用户明确要求忘记时，MemoryAgent 可以删除精确匹配的旧 bullet。"""
    user_file = tmp_path / "USER.md"
    user_file.write_text("# User Memory\n\n## Preferences\n\n- Uses Python.\n", encoding="utf-8")
    session = Session(key="web:default")
    session.add_message("user", "忘记我使用 Python。")
    session.add_message("assistant", "好的。")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "should_update": True,
            "operations": [{
                "target": "user",
                "op": "delete",
                "section": "Preferences",
                "match": "- Uses Python.",
                "content": "",
                "confidence": "high",
                "reason": "User explicitly asked to forget it.",
            }],
            "reason": "User requested deletion.",
        })
    ))
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    updated = await extractor.extract_new(session)

    assert updated is True
    assert "Uses Python" not in user_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_memory_agent_skips_low_confidence_user_memory(tmp_path: Path) -> None:
    """低置信度自动用户记忆不会写入 USER.md。"""
    session = Session(key="web:default")
    session.add_message("user", "这次详细解释一下。")
    session.add_message("assistant", "好的。")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "should_update": True,
            "operations": [{
                "target": "user",
                "op": "append",
                "section": "Preferences",
                "match": "",
                "content": "- Prefers detailed explanations.",
                "confidence": "low",
                "reason": "This is likely temporary.",
            }],
            "reason": "Low confidence.",
        })
    ))
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    updated = await extractor.extract_new(session)

    assert updated is False
    assert not user_file_exists(tmp_path)


@pytest.mark.asyncio
async def test_memory_agent_skips_when_explicit_memory_tool_was_used(tmp_path: Path) -> None:
    """当前轮已经显式写记忆时，后台提取不重复扫描。"""
    session = Session(key="web:default")
    session.add_message("user", "记住我使用 Python。")
    session.messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "manage_memory", "arguments": "{}"},
        }],
    })
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock()
    extractor = MemoryAgent(tmp_path, provider, "test-model")

    with patch("minicrab.agent.memory_extractor.logger.info") as info_log:
        updated = await extractor.extract_new(session)

    assert updated is False
    provider.chat_with_retry.assert_not_awaited()
    assert session.metadata["memory_extracted_until"] == len(session.messages)
    assert any("explicit memory tool already used" in str(call) for call in info_log.call_args_list)


def user_file_exists(tmp_path: Path) -> bool:
    """判断测试工作区是否生成 USER.md。"""
    return (tmp_path / "USER.md").exists()
