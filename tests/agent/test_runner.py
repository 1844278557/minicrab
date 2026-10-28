"""Tests for the shared agent runner and its integration contracts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minicrab.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path):
    from minicrab.agent.loop import AgentLoop
    from minicrab.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with patch("minicrab.agent.loop.ContextBuilder"), \
         patch("minicrab.agent.loop.SessionManager"), \
         patch("minicrab.agent.loop.SubagentManager") as MockSubMgr:
        MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path)
    return loop


@pytest.mark.asyncio
async def test_runner_preserves_reasoning_fields_and_tool_results():
    from minicrab.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    captured_second_call: list[dict] = []
    call_count = {"n": 0}

    async def chat_with_retry(*, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
                reasoning_content="hidden reasoning",
                thinking_blocks=[{"type": "thinking", "thinking": "step"}],
                usage={"prompt_tokens": 5, "completion_tokens": 3},
            )
        captured_second_call[:] = messages
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=3,
    ))

    assert result.final_content == "done"
    assert result.tools_used == ["list_dir"]
    assert result.tool_events == [
        {"name": "list_dir", "status": "ok", "detail": "tool result"}
    ]

    assistant_messages = [
        msg for msg in captured_second_call
        if msg.get("role") == "assistant" and msg.get("tool_calls")
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["reasoning_content"] == "hidden reasoning"
    assert assistant_messages[0]["thinking_blocks"] == [{"type": "thinking", "thinking": "step"}]
    assert any(
        msg.get("role") == "tool" and msg.get("content") == "tool result"
        for msg in captured_second_call
    )


@pytest.mark.asyncio
async def test_runner_summarizes_long_tool_result_without_tools():
    from minicrab.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    calls: list[dict] = []
    captured_final_messages: list[dict] = []
    long_result = "HEAD-" + ("x" * 13_000) + "-TAIL"

    async def chat_with_retry(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return LLMResponse(
                content="working",
                tool_calls=[ToolCallRequest(id="call_1", name="exec", arguments={"command": "pytest"})],
            )
        if len(calls) == 2:
            return LLMResponse(content="- 测试失败摘要\n- 关键错误在尾部")
        captured_final_messages[:] = kwargs["messages"]
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value=long_result)

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=3,
    ))

    assert result.final_content == "done"
    summary_call = calls[1]
    assert summary_call["tools"] == []
    assert summary_call["tool_choice"] == "none"
    assert summary_call["max_tokens"] == 700
    assert summary_call["temperature"] == 0

    tool_messages = [m for m in captured_final_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    content = tool_messages[0]["content"]
    assert "[LLM Tool Result Summary]" in content
    assert "测试失败摘要" in content
    assert "[Original Head]" in content
    assert "[Original Tail]" in content
    assert len(content) < len(long_result)


@pytest.mark.asyncio
async def test_runner_falls_back_when_tool_result_summary_fails():
    from minicrab.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    call_count = {"n": 0}
    captured_final_messages: list[dict] = []
    long_result = "HEAD-" + ("x" * 13_000) + "-TAIL"

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="working",
                tool_calls=[ToolCallRequest(id="call_1", name="exec", arguments={"command": "pytest"})],
            )
        if call_count["n"] == 2:
            return LLMResponse(content="summary failed", finish_reason="error")
        captured_final_messages[:] = kwargs["messages"]
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value=long_result)

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=3,
    ))

    assert result.final_content == "done"
    tool_messages = [m for m in captured_final_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    content = tool_messages[0]["content"]
    assert "[Tool Result Truncated]" in content
    assert "LLM summary unavailable" in content
    assert "[Original Head]" in content
    assert "[Original Tail]" in content


@pytest.mark.asyncio
async def test_runner_calls_hooks_in_order():
    from minicrab.agent.hook import AgentHook, AgentHookContext
    from minicrab.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    call_count = {"n": 0}
    events: list[tuple] = []

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
            )
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    class RecordingHook(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            events.append(("before_iteration", context.iteration))

        async def before_execute_tools(self, context: AgentHookContext) -> None:
            events.append((
                "before_execute_tools",
                context.iteration,
                [tc.name for tc in context.tool_calls],
            ))

        async def after_iteration(self, context: AgentHookContext) -> None:
            events.append((
                "after_iteration",
                context.iteration,
                context.final_content,
                list(context.tool_results),
                list(context.tool_events),
                context.stop_reason,
            ))

        def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
            events.append(("finalize_content", context.iteration, content))
            return content.upper() if content else content

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=3,
        hook=RecordingHook(),
    ))

    assert result.final_content == "DONE"
    assert events == [
        ("before_iteration", 0),
        ("before_execute_tools", 0, ["list_dir"]),
        (
            "after_iteration",
            0,
            None,
            ["tool result"],
            [{"name": "list_dir", "status": "ok", "detail": "tool result"}],
            None,
        ),
        ("before_iteration", 1),
        ("finalize_content", 1, "done"),
        ("after_iteration", 1, "DONE", [], [], "completed"),
    ]


@pytest.mark.asyncio
async def test_runner_streaming_hook_receives_deltas_and_end_signal():
    from minicrab.agent.hook import AgentHook, AgentHookContext
    from minicrab.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    streamed: list[str] = []
    endings: list[bool] = []

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        await on_content_delta("he")
        await on_content_delta("llo")
        return LLMResponse(content="hello", tool_calls=[], usage={})

    provider.chat_stream_with_retry = chat_stream_with_retry
    provider.chat_with_retry = AsyncMock()
    tools = MagicMock()
    tools.get_definitions.return_value = []

    class StreamingHook(AgentHook):
        def wants_streaming(self) -> bool:
            return True

        async def on_stream(self, context: AgentHookContext, delta: str) -> None:
            streamed.append(delta)

        async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
            endings.append(resuming)

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=1,
        hook=StreamingHook(),
    ))

    assert result.final_content == "hello"
    assert streamed == ["he", "llo"]
    assert endings == [False]
    provider.chat_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_returns_max_iterations_fallback():
    from minicrab.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="still working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
    ))

    assert result.stop_reason == "max_iterations"
    assert result.final_content == (
        "I reached the maximum number of tool call iterations (2) "
        "without completing the task. You can try breaking the task into smaller steps."
    )


@pytest.mark.asyncio
async def test_runner_returns_structured_tool_error():
    from minicrab.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(side_effect=RuntimeError("boom"))

    runner = AgentRunner(provider)

    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
        fail_on_tool_error=True,
    ))

    assert result.stop_reason == "tool_error"
    assert result.error == "Error: RuntimeError: boom"
    assert result.tool_events == [
        {"name": "list_dir", "status": "error", "detail": "boom"}
    ]


@pytest.mark.asyncio
async def test_runner_parallelizes_only_safe_tool_groups():
    """只读工具并行执行，写入和消息类工具保持顺序。"""
    from minicrab.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    call_count = {"n": 0}
    final_messages: list[dict] = []

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="working",
                tool_calls=[
                    ToolCallRequest(id="call_1", name="read_file", arguments={"path": "a.md"}),
                    ToolCallRequest(id="call_2", name="list_dir", arguments={"path": "."}),
                    ToolCallRequest(id="call_3", name="write_file", arguments={"path": "a.md", "content": "x"}),
                    ToolCallRequest(id="call_4", name="search_rag", arguments={"query": "x"}),
                    ToolCallRequest(id="call_5", name="list_rag_documents", arguments={}),
                    ToolCallRequest(id="call_6", name="message", arguments={"content": "done"}),
                ],
            )
        final_messages[:] = kwargs["messages"]
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    running: set[str] = set()
    events: list[tuple[str, str, tuple[str, ...]]] = []
    safe_overlap = {"seen": False}
    unsafe_overlap = {"seen": False}

    async def execute(name: str, arguments: dict):
        running.add(name)
        events.append(("start", name, tuple(sorted(running))))
        if name in {"read_file", "list_dir"} and {"read_file", "list_dir"} <= running:
            safe_overlap["seen"] = True
        if name in {"write_file", "message"} and len(running) > 1:
            unsafe_overlap["seen"] = True
        await asyncio.sleep(0.01)
        events.append(("finish", name, tuple(sorted(running))))
        running.remove(name)
        return f"{name} result"

    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = execute

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
        concurrent_tools=True,
    ))

    assert result.final_content == "done"
    assert safe_overlap["seen"] is True
    assert unsafe_overlap["seen"] is False
    write_start = next(
        index for index, event in enumerate(events)
        if event[0] == "start" and event[1] == "write_file"
    )
    read_finish = next(
        index for index, event in enumerate(events)
        if event[0] == "finish" and event[1] == "read_file"
    )
    list_finish = next(
        index for index, event in enumerate(events)
        if event[0] == "finish" and event[1] == "list_dir"
    )
    assert read_finish < write_start
    assert list_finish < write_start
    tool_messages = [message for message in final_messages if message.get("role") == "tool"]
    assert [message["name"] for message in tool_messages] == [
        "read_file",
        "list_dir",
        "write_file",
        "search_rag",
        "list_rag_documents",
        "message",
    ]


@pytest.mark.asyncio
async def test_loop_max_iterations_message_stays_stable(tmp_path):
    loop = _make_loop(tmp_path)
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
    ))
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="ok")
    loop.max_iterations = 2

    final_content, _, _ = await loop._run_agent_loop([])

    assert final_content == (
        "I reached the maximum number of tool call iterations (2) "
        "without completing the task. You can try breaking the task into smaller steps."
    )


@pytest.mark.asyncio
async def test_loop_stream_filter_handles_think_only_prefix_without_crashing(tmp_path):
    loop = _make_loop(tmp_path)
    deltas: list[str] = []
    endings: list[bool] = []

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        await on_content_delta("<think>hidden")
        await on_content_delta("</think>Hello")
        return LLMResponse(content="<think>hidden</think>Hello", tool_calls=[], usage={})

    loop.provider.chat_stream_with_retry = chat_stream_with_retry

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    async def on_stream_end(*, resuming: bool = False) -> None:
        endings.append(resuming)

    final_content, _, _ = await loop._run_agent_loop(
        [],
        on_stream=on_stream,
        on_stream_end=on_stream_end,
    )

    assert final_content == "Hello"
    assert deltas == ["Hello"]
    assert endings == [False]


@pytest.mark.asyncio
async def test_subagent_max_iterations_announces_existing_fallback(tmp_path, monkeypatch):
    from minicrab.agent.subagent import SubagentManager
    from minicrab.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
    ))
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)
    mgr._announce_result = AsyncMock()

    async def fake_execute(self, name, arguments):
        return "tool result"

    monkeypatch.setattr("minicrab.agent.tools.registry.ToolRegistry.execute", fake_execute)

    await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})

    mgr._announce_result.assert_awaited_once()
    args = mgr._announce_result.await_args.args
    assert args[3] == "Task completed but no final response was generated."
    assert args[5] == "ok"
