"""测试人工审核的最小执行流程。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest


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
                arguments={"content": "# User Profile\n\n- 女朋友生日是 5 月 20 日。"},
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
            "content": "需要你确认：\n\n编号：A-test\n操作：写入长期用户记忆",
            "approval_id": "A-test",
            "tool_name": "remember_user_info",
            "summary": "写入长期用户记忆",
        },
    ))

    assert result.stop_reason == "approval_required"
    assert result.final_content is None
    assert result.approval_event is not None
    assert result.approval_event["approval_id"] == "A-test"
    assert result.messages == []
    tools.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_loop_approval_executes_remember_user_info_after_confirm(tmp_path) -> None:
    """用户同意后，原工具才会执行。"""
    provider = _make_provider()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="remember_user_info",
                    arguments={"content": "# User Profile\n\n- 女朋友生日是 5 月 20 日。"},
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
    assert not (tmp_path / "USER.md").exists()

    second = await loop._process_message(
        InboundMessage(channel="web", sender_id="user", chat_id="default", content="同意")
    )

    assert second is not None
    assert second.content == "已记住。"
    assert "女朋友生日是 5 月 20 日" in (tmp_path / "USER.md").read_text(encoding="utf-8")
    history = loop.get_session_history("web:default", 100)
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert all("需要你确认" not in message["content"] for message in history)
    assert all("同意" not in message["content"] for message in history)


@pytest.mark.asyncio
async def test_loop_approval_reject_does_not_execute_tool(tmp_path) -> None:
    """用户拒绝后，原工具不会执行。"""
    provider = _make_provider()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="remember_user_info",
                    arguments={"content": "# User Profile\n\n- 不应写入。"},
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
    assert not (tmp_path / "USER.md").exists()
