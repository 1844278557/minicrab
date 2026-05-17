"""测试 Web 使用的服务端会话历史过滤。"""

from __future__ import annotations

from unittest.mock import MagicMock

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings


def _make_loop(tmp_path) -> AgentLoop:
    """创建带真实临时工作区的 AgentLoop。"""
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.estimate_prompt_tokens.return_value = (0, "test-counter")
    return AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")


def test_get_session_history_filters_internal_messages(tmp_path) -> None:
    """Web 历史只返回用户和助手的可展示文本。"""
    loop = _make_loop(tmp_path)
    session = loop.sessions.get_or_create("web:default")
    session.messages = [
        {"role": "system", "content": "system", "timestamp": "1"},
        {"role": "user", "content": "你好", "timestamp": "2"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}], "timestamp": "3"},
        {"role": "tool", "content": "tool result", "timestamp": "4"},
        {"role": "assistant", "content": "你好，有什么需要？", "timestamp": "5"},
        {"role": "assistant", "content": "   ", "timestamp": "6"},
        {
            "role": "assistant",
            "content": (
                "需要你确认：\n\n"
                "编号：A-20260517215805-001\n"
                "操作：写入长期用户记忆\n"
                "原因：会把信息长期保存到 USER.md。\n\n"
                "回复：\n"
                "同意 A-20260517215805-001\n"
                "拒绝 A-20260517215805-001"
            ),
            "timestamp": "7",
        },
    ]

    history = loop.get_session_history("web:default", limit=100)

    assert history == [
        {"role": "user", "content": "你好", "timestamp": "2"},
        {"role": "assistant", "content": "你好，有什么需要？", "timestamp": "5"},
    ]


def test_get_session_history_caps_to_recent_100_visible_messages(tmp_path) -> None:
    """Web 历史最多返回最近 100 条可展示消息。"""
    loop = _make_loop(tmp_path)
    session = loop.sessions.get_or_create("web:default")
    session.messages = [
        {"role": "user", "content": f"m{i}", "timestamp": str(i)}
        for i in range(120)
    ]

    history = loop.get_session_history("web:default", limit=1000)

    assert len(history) == 100
    assert history[0]["content"] == "m20"
    assert history[-1]["content"] == "m119"


def test_get_session_history_zero_limit_returns_empty_list(tmp_path) -> None:
    """limit 为 0 时不返回历史。"""
    loop = _make_loop(tmp_path)
    session = loop.sessions.get_or_create("web:default")
    session.messages = [{"role": "user", "content": "hello", "timestamp": "1"}]

    assert loop.get_session_history("web:default", limit=0) == []


def test_get_session_history_extracts_text_blocks(tmp_path) -> None:
    """数组内容只提取文本块，忽略图片等非文本内容。"""
    loop = _make_loop(tmp_path)
    session = loop.sessions.get_or_create("web:default")
    session.messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                {"type": "text", "text": "看这张图"},
            ],
            "timestamp": "1",
        }
    ]

    history = loop.get_session_history("web:default")

    assert history == [{"role": "user", "content": "看这张图", "timestamp": "1"}]
