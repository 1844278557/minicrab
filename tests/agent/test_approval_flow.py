"""测试人工审核的最小执行流程。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from minicrab.agent.loop import AgentLoop
from minicrab.agent.runner import AgentRunner, AgentRunSpec
from minicrab.agent.tools.message import MessageTool
from minicrab.bus.events import InboundMessage
from minicrab.bus.queue import MessageBus
from minicrab.providers.base import GenerationSettings, LLMResponse, ToolCallRequest


def _make_provider() -> MagicMock:
    """创建测试用模型提供者。"""
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.estimate_prompt_tokens.return_value = (0, "test-counter")
    return provider


@pytest.mark.asyncio
async def test_runner_stops_before_approved_tool_executes() -> None:
    """需要审核时，Runner 不会直接执行工具。"""
    provider = _make_provider()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="",
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="remember_user_info",
                arguments={
                    "content": "# Working Memory\n\n- Web 审批使用弹窗。",
                    "target": "memory",
                },
            )
        ],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="should not run")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=3,
        approval_callback=lambda _tool_call: {
            "content": "需要你确认：\n\n编号：A-test\n操作：写入长期工作记忆",
            "approval_id": "A-test",
            "tool_name": "remember_user_info",
            "summary": "写入长期工作记忆",
        },
    ))

    assert result.stop_reason == "approval_required"
    assert result.final_content is None
    assert result.approval_event is not None
    assert result.approval_event["approval_id"] == "A-test"
    assert result.messages == []
    tools.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_denied_tool_returns_permission_error_without_execution() -> None:
    """权限拒绝时，Runner 不执行工具，并把错误结果交给模型。"""
    provider = _make_provider()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="exec",
                    arguments={"command": "format C:"},
                )
            ],
        ),
        LLMResponse(content="这个操作被权限系统拒绝了。", tool_calls=[]),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="should not run")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=3,
        approval_callback=lambda tool_call, *_args: {
            "status": "denied",
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "summary": "执行命令",
            "reason": "会修改磁盘或文件系统。",
        },
    ))

    assert result.stop_reason == "completed"
    assert result.final_content == "这个操作被权限系统拒绝了。"
    assert result.tool_events[0]["status"] == "permission_denied"
    assert "Permission denied" in result.messages[1]["content"]
    tools.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_loop_approval_executes_working_memory_after_confirm(tmp_path) -> None:
    """用户同意后，长期工作记忆工具才会执行。"""
    provider = _make_provider()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="remember_user_info",
                    arguments={
                        "content": "# Working Memory\n\n- Web 审批使用弹窗。",
                        "target": "memory",
                    },
                )
            ],
        ),
        LLMResponse(content="已记住。", tool_calls=[]),
    ])
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop.tools.get_definitions = MagicMock(return_value=[])

    first = await loop._process_message(
        InboundMessage(channel="web", sender_id="user", chat_id="default", content="帮我记住")
    )

    assert first is not None
    assert first.content == ""
    assert first.metadata["_control"] == "approval_required"
    assert first.metadata["approval"]["id"].startswith("A-")
    assert not (tmp_path / "memory" / "MEMORY.md").exists()

    second = await loop._process_message(
        InboundMessage(channel="web", sender_id="user", chat_id="default", content="同意")
    )

    assert second is not None
    assert second.content == "已记住。"
    assert "Web 审批使用弹窗" in (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    history = loop.get_session_history("web:default", 100)
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert all("需要你确认" not in message["content"] for message in history)
    assert all("同意" not in message["content"] for message in history)


@pytest.mark.asyncio
async def test_loop_user_memory_write_does_not_require_approval(tmp_path) -> None:
    """USER.md 写入不再弹出人工确认。"""
    provider = _make_provider()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="remember_user_info",
                    arguments={"content": "# User Profile\n\n- 使用 Python。"},
                )
            ],
        ),
        LLMResponse(content="已记住。", tool_calls=[]),
    ])
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop.tools.get_definitions = MagicMock(return_value=[])

    response = await loop._process_message(
        InboundMessage(channel="web", sender_id="user", chat_id="default", content="记住我使用 Python")
    )

    assert response is not None
    assert response.content == "已记住。"
    assert response.metadata.get("_control") != "approval_required"
    assert "使用 Python" in (tmp_path / "USER.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_loop_approval_message_tool_response_consumes_confirmation(tmp_path) -> None:
    """审批后如果 message 工具已发消息，确认文本不能再进入普通对话。"""
    provider = _make_provider()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="remember_user_info",
                    arguments={"content": "# Working Memory\n\n- 使用 Python。", "target": "memory"},
                )
            ],
        ),
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="call_2",
                    name="message",
                    arguments={"content": "已记住。"},
                )
            ],
        ),
        LLMResponse(content="内部结束。", tool_calls=[]),
    ])
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop.tools.get_definitions = MagicMock(return_value=[])
    sent_messages = []

    async def collect_message(message):
        sent_messages.append(message)

    message_tool = loop.tools.get("message")
    assert isinstance(message_tool, MessageTool)
    message_tool.set_send_callback(collect_message)

    await loop._process_message(
        InboundMessage(channel="web", sender_id="user", chat_id="default", content="记住我使用 Python")
    )
    response = await loop._process_message(
        InboundMessage(channel="web", sender_id="user", chat_id="default", content="同意")
    )

    assert response is None
    assert [message.content for message in sent_messages] == ["已记住。"]
    assert provider.chat_with_retry.await_count == 2
    history = loop.get_session_history("web:default", 100)
    assert not any(message["role"] == "user" and message["content"] == "同意" for message in history)
    assert not any("内部结束" in message["content"] for message in history)


@pytest.mark.asyncio
async def test_loop_approval_reject_does_not_execute_tool(tmp_path) -> None:
    """用户拒绝后，长期工作记忆工具不会执行。"""
    provider = _make_provider()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="remember_user_info",
                    arguments={"content": "# Working Memory\n\n- 不应写入。", "target": "memory"},
                )
            ],
        ),
        LLMResponse(content="已取消。", tool_calls=[]),
    ])
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop.tools.get_definitions = MagicMock(return_value=[])

    await loop._process_message(
        InboundMessage(channel="web", sender_id="user", chat_id="default", content="帮我记住")
    )
    response = await loop._process_message(
        InboundMessage(channel="web", sender_id="user", chat_id="default", content="拒绝")
    )

    assert response is not None
    assert response.content == "已取消。"
    assert not (tmp_path / "memory" / "MEMORY.md").exists()


@pytest.mark.asyncio
async def test_loop_reject_suppresses_inline_tool_call_markup(tmp_path) -> None:
    """拒绝后模型写出的伪工具调用正文不能展示给用户。"""
    provider = _make_provider()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="cleanup_expired_rag",
                    arguments={},
                )
            ],
        ),
        LLMResponse(
            content=(
                "让我继续查看数据库。\n\n"
                "<｜DSML｜function_calls>\n"
                "<｜DSML｜invoke name=\"exec\">\n"
                "<｜DSML｜parameter name=\"command\" string=\"true\">sqlite3 chroma.sqlite3</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜function_calls>"
            ),
            tool_calls=[],
        ),
    ])
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop.tools.get_definitions = MagicMock(return_value=[])

    await loop._process_message(
        InboundMessage(channel="web", sender_id="user", chat_id="default", content="查看一下数据库")
    )
    response = await loop._process_message(
        InboundMessage(channel="web", sender_id="user", chat_id="default", content="拒绝")
    )

    assert response is not None
    assert response.content == "操作已取消，未执行。"
    assert "<｜DSML｜function_calls>" not in response.content
    history = loop.get_session_history("web:default", 100)
    assert not any("DSML" in message["content"] for message in history)


@pytest.mark.asyncio
async def test_loop_reject_does_not_allow_alternate_working_memory_write(tmp_path) -> None:
    """用户拒绝工作记忆写入后，模型不能换用 edit_file 写 MEMORY.md。"""
    provider = _make_provider()
    second_call_tools: list = []

    async def chat_with_retry(*, tools, **_kwargs):
        if not second_call_tools:
            second_call_tools.append(None)
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="remember_user_info",
                        arguments={"content": "# Working Memory\n\n- 使用 Python。", "target": "memory"},
                    )
                ],
            )
        second_call_tools[0] = tools
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="call_2",
                    name="edit_file",
                    arguments={
                        "path": str(tmp_path / "memory" / "MEMORY.md"),
                        "old_text": "- ",
                        "new_text": "- 使用 Python。",
                    },
                )
            ],
        )

    provider.chat_with_retry = AsyncMock(side_effect=chat_with_retry)
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop.tools.get_definitions = MagicMock(return_value=[{"type": "function", "function": {"name": "edit_file"}}])

    await loop._process_message(
        InboundMessage(channel="web", sender_id="user", chat_id="default", content="记住我使用 Python")
    )
    response = await loop._process_message(
        InboundMessage(channel="web", sender_id="user", chat_id="default", content="拒绝")
    )

    assert response is not None
    assert response.content == "操作已取消，未执行。"
    assert second_call_tools[0] == []
    assert not (tmp_path / "memory" / "MEMORY.md").exists()
