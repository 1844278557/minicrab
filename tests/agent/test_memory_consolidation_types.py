"""测试会话压缩只生成 running summary，不写长期记忆或 episode。"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from minicrab.agent.memory import MemoryStore
from minicrab.providers.base import LLMProvider, LLMResponse, ToolCallRequest


def _make_messages(message_count: int = 30):
    """创建测试消息。"""
    return [
        {"role": "user", "content": f"msg{i}", "timestamp": "2026-01-01 00:00"}
        for i in range(message_count)
    ]


def _make_tool_response(summary):
    """创建 save_session_summary 工具响应。"""
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="save_session_summary",
                arguments={
                    "topic": "Testing",
                    "summary": summary,
                },
            )
        ],
    )


def _assert_no_long_term_files(store: MemoryStore) -> None:
    """压缩不能写长期记忆、历史文件或 episode。"""
    assert not store.history_file.exists()
    assert not store.memory_file.exists()
    assert not any(store.episode_store.episodes_dir.glob("*.json"))


class ScriptedProvider(LLMProvider):
    """按顺序返回预设 LLM 响应。"""

    def __init__(self, responses: list[LLMResponse]):
        super().__init__()
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, *args, **kwargs) -> LLMResponse:
        """返回下一条响应。"""
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        """返回测试模型名。"""
        return "test-model"


class TestSessionCompactionTypeHandling:
    """测试会话压缩对各种参数类型的处理。"""

    @pytest.mark.asyncio
    async def test_string_summary_returns_session_summary(self, tmp_path: Path) -> None:
        """正常情况返回字符串摘要，不写长期文件。"""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=_make_tool_response("User discussed testing.")
        )

        result = await store.consolidate(_make_messages(60), provider, "test-model")

        assert result == "User discussed testing."
        _assert_no_long_term_files(store)

    @pytest.mark.asyncio
    async def test_dict_summary_serialized_to_json(self, tmp_path: Path) -> None:
        """LLM 返回 dict 摘要时应序列化，避免 TypeError。"""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=_make_tool_response({"summary": "User discussed testing."})
        )

        result = await store.consolidate(_make_messages(60), provider, "test-model")

        parsed = json.loads(result)
        assert parsed["summary"] == "User discussed testing."
        _assert_no_long_term_files(store)

    @pytest.mark.asyncio
    async def test_arguments_can_be_raw_json_string(self, tmp_path: Path) -> None:
        """部分 provider 会把工具参数作为 JSON 字符串返回。"""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_session_summary",
                    arguments=json.dumps({
                        "topic": "Testing",
                        "summary": "Raw JSON summary.",
                    }),
                )
            ],
        ))

        result = await store.consolidate(_make_messages(60), provider, "test-model")

        assert result == "Raw JSON summary."
        _assert_no_long_term_files(store)

    @pytest.mark.asyncio
    async def test_list_arguments_extract_first_dict(self, tmp_path: Path) -> None:
        """列表参数里第一个元素是 dict 时可以正常处理。"""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_session_summary",
                    arguments=[{"summary": "List summary."}],
                )
            ],
        ))

        result = await store.consolidate(_make_messages(60), provider, "test-model")

        assert result == "List summary."
        _assert_no_long_term_files(store)

    @pytest.mark.asyncio
    async def test_empty_messages_return_existing_summary(self, tmp_path: Path) -> None:
        """没有要压缩的消息时直接返回已有摘要。"""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock()

        result = await store.consolidate([], provider, "test-model", current_summary="Existing")

        assert result == "Existing"
        provider.chat_with_retry.assert_not_called()
        _assert_no_long_term_files(store)

    @pytest.mark.asyncio
    async def test_invalid_arguments_return_none_without_writing(self, tmp_path: Path) -> None:
        """空列表、非 dict 列表、缺失摘要都不写任何长期文件。"""
        cases = [
            [],
            ["bad"],
            {"topic": "Missing summary"},
            {"summary": None},
            {"summary": "   "},
        ]
        for arguments in cases:
            store = MemoryStore(tmp_path / str(len(str(arguments))))
            provider = AsyncMock()
            provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="save_session_summary",
                        arguments=arguments,
                    )
                ],
            ))

            result = await store.consolidate(_make_messages(10), provider, "test-model")

            assert result is None
            _assert_no_long_term_files(store)

    @pytest.mark.asyncio
    async def test_no_tool_call_returns_none(self, tmp_path: Path) -> None:
        """没有工具调用时返回 None。"""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(content="I summarized it.", tool_calls=[])
        )

        result = await store.consolidate(_make_messages(60), provider, "test-model")

        assert result is None
        _assert_no_long_term_files(store)

    @pytest.mark.asyncio
    async def test_retries_transient_error_then_succeeds(self, tmp_path: Path, monkeypatch) -> None:
        """Provider 的瞬时错误仍由 chat_with_retry 处理。"""
        store = MemoryStore(tmp_path)
        provider = ScriptedProvider([
            LLMResponse(content="503 server error", finish_reason="error"),
            _make_tool_response("Recovered summary."),
        ])
        delays: list[int] = []

        async def _fake_sleep(delay: int) -> None:
            delays.append(delay)

        monkeypatch.setattr("minicrab.providers.base.asyncio.sleep", _fake_sleep)

        result = await store.consolidate(_make_messages(60), provider, "test-model")

        assert result == "Recovered summary."
        assert provider.calls == 2
        assert delays == [1]
        _assert_no_long_term_files(store)

    @pytest.mark.asyncio
    async def test_consolidation_delegates_to_provider_defaults(self, tmp_path: Path) -> None:
        """压缩调用仍由 provider 管理默认生成参数。"""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=_make_tool_response("Provider defaults summary.")
        )

        result = await store.consolidate(_make_messages(60), provider, "test-model")

        assert result == "Provider defaults summary."
        provider.chat_with_retry.assert_awaited_once()
        _, kwargs = provider.chat_with_retry.await_args
        assert kwargs["model"] == "test-model"
        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs
        assert "reasoning_effort" not in kwargs

    @pytest.mark.asyncio
    async def test_tool_choice_fallback_on_unsupported_error(self, tmp_path: Path) -> None:
        """强制 tool_choice 不支持时，退回 auto 后仍可成功。"""
        store = MemoryStore(tmp_path)
        error_resp = LLMResponse(
            content="Error: The tool_choice parameter does not support being set to object",
            finish_reason="error",
            tool_calls=[],
        )
        ok_resp = _make_tool_response("Fallback summary.")
        call_log: list[dict] = []

        async def _tracking_chat(**kwargs):
            call_log.append(kwargs)
            return error_resp if len(call_log) == 1 else ok_resp

        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(side_effect=_tracking_chat)

        result = await store.consolidate(_make_messages(60), provider, "test-model")

        assert result == "Fallback summary."
        assert isinstance(call_log[0]["tool_choice"], dict)
        assert call_log[1]["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_tool_choice_fallback_auto_no_tool_call(self, tmp_path: Path) -> None:
        """auto 重试后仍无工具调用时返回 None。"""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(side_effect=[
            LLMResponse(content="Error: tool_choice must be none or auto", finish_reason="error"),
            LLMResponse(content="No tool.", finish_reason="stop", tool_calls=[]),
        ])

        result = await store.consolidate(_make_messages(60), provider, "test-model")

        assert result is None
        _assert_no_long_term_files(store)

    @pytest.mark.asyncio
    async def test_raw_summary_after_consecutive_failures(self, tmp_path: Path) -> None:
        """连续失败后返回确定性兜底摘要，但仍不写长期文件或 episode。"""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(content="No tool call.", finish_reason="stop", tool_calls=[])
        )
        messages = _make_messages(10)

        assert await store.consolidate(messages, provider, "m") is None
        assert await store.consolidate(messages, provider, "m") is None
        result = await store.consolidate(messages, provider, "m")

        assert "Compacted Session Chunk" in result
        assert "10 messages" in result
        assert "msg0" in result
        _assert_no_long_term_files(store)

    @pytest.mark.asyncio
    async def test_raw_summary_counter_resets_on_success(self, tmp_path: Path) -> None:
        """成功压缩会重置失败计数。"""
        store = MemoryStore(tmp_path)
        no_tool = LLMResponse(content="Nope.", finish_reason="stop", tool_calls=[])
        ok_resp = _make_tool_response("OK summary.")
        messages = _make_messages(10)

        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(return_value=no_tool)
        assert await store.consolidate(messages, provider, "m") is None
        assert await store.consolidate(messages, provider, "m") is None
        assert store._consecutive_failures == 2

        provider.chat_with_retry = AsyncMock(return_value=ok_resp)
        assert await store.consolidate(messages, provider, "m") == "OK summary."
        assert store._consecutive_failures == 0

        provider.chat_with_retry = AsyncMock(return_value=no_tool)
        assert await store.consolidate(messages, provider, "m") is None
        assert store._consecutive_failures == 1
