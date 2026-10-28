"""Tests for cache-friendly prompt construction."""

from __future__ import annotations

from datetime import datetime as real_datetime
from importlib.resources import files as pkg_files
from pathlib import Path
import datetime as datetime_module

from minicrab.agent.context import ContextBuilder


class _FakeDatetime(real_datetime):
    current = real_datetime(2026, 2, 24, 13, 59)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls.current


def _make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return workspace


def test_bootstrap_files_are_backed_by_templates() -> None:
    template_dir = pkg_files("minicrab") / "templates"

    for filename in ContextBuilder.BOOTSTRAP_FILES:
        assert (template_dir / filename).is_file(), f"missing bootstrap template: {filename}"


def test_context_template_uses_structured_task_state() -> None:
    """CONTEXT.md 模板应保留稳定任务状态字段。"""
    template_dir = pkg_files("minicrab") / "templates"
    content = (template_dir / "CONTEXT.md").read_text(encoding="utf-8")

    for heading in (
        "## Current Goal",
        "## Active Step",
        "## Completed Steps",
        "## Next Step",
        "## Blockers",
        "## Key Decisions",
        "## Validation",
        "## Cautions",
    ):
        assert heading in content


def test_system_prompt_stays_stable_when_clock_changes(tmp_path, monkeypatch) -> None:
    """System prompt should not change just because wall clock minute changes."""
    monkeypatch.setattr(datetime_module, "datetime", _FakeDatetime)

    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    _FakeDatetime.current = real_datetime(2026, 2, 24, 13, 59)
    prompt1 = builder.build_system_prompt()

    _FakeDatetime.current = real_datetime(2026, 2, 24, 14, 0)
    prompt2 = builder.build_system_prompt()

    assert prompt1 == prompt2


def test_system_prompt_contains_reminder_rules(tmp_path) -> None:
    """系统提示应约束提醒任务直接调用工具且不展示工具语法。"""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace, timezone="Asia/Shanghai")

    prompt = builder.build_system_prompt()

    assert "## Reminder Guidelines" in prompt
    assert "Do not pass `tz` together with one-time `at`" in prompt
    assert "Do not show tool syntax such as `cron.add(...)`" in prompt


def test_system_prompt_contains_memory_routing_rules(tmp_path) -> None:
    """系统提示应明确不同记忆内容的存放位置。"""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    prompt = builder.build_system_prompt()

    assert "## Memory Routing Guidelines" in prompt
    assert "USER.md is for stable user identity, preferences" in prompt
    assert "CONTEXT.md is for the current local project state only" in prompt
    assert "Current Goal, Active Step, Completed Steps, Next Step, Blockers, Key Decisions, Validation, and Cautions" in prompt
    assert "MEMORY.md is for durable cross-session experience" in prompt
    assert "RAG is for uploaded documents and reference files" in prompt
    assert "Episodes are a legacy archive of old conversations" in prompt
    assert "Do not store temporary chat, raw tool output, document contents, or current task progress in USER.md or MEMORY.md" in prompt


def test_context_md_is_loaded_as_task_state_block(tmp_path) -> None:
    """CONTEXT.md 应作为当前任务摘要块进入系统提示。"""
    workspace = _make_workspace(tmp_path)
    (workspace / "CONTEXT.md").write_text(
        "# Current Task State\n\n## Current Goal\n\nShip memory routing.",
        encoding="utf-8",
    )
    builder = ContextBuilder(workspace)

    prompt = builder.build_system_prompt()

    assert "## CONTEXT.md" in prompt
    assert "Ship memory routing." in prompt
    assert "Current task state:" in prompt


def test_long_term_memory_is_truncated_by_budget(tmp_path, monkeypatch) -> None:
    """自动长期记忆过长时应按预算截断，避免挤占上下文。"""
    workspace = _make_workspace(tmp_path)
    memory_dir = workspace / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("old-content-" + "x" * 100 + "recent-tail", encoding="utf-8")
    monkeypatch.setattr(ContextBuilder, "MAX_LONG_TERM_MEMORY_CHARS", 20)
    builder = ContextBuilder(workspace)

    prompt = builder.build_system_prompt()

    assert "[Long-term memory truncated to fit context budget]" in prompt
    assert "recent-tail" in prompt
    assert "old-content" not in prompt


def test_session_summary_is_loaded_into_system_message(tmp_path) -> None:
    """压缩后的当前会话摘要应进入系统消息。"""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    messages = builder.build_messages(
        history=[],
        current_message="continue",
        session_summary="Older context: user already chose Web modal approval.",
    )

    system_content = messages[0]["content"]
    assert "# Current Session Summary" in system_content
    assert "Older context: user already chose Web modal approval." in system_content


def test_session_harness_state_is_loaded_into_system_message(tmp_path) -> None:
    """会话框架状态应进入系统消息，帮助压缩后恢复上下文。"""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    messages = builder.build_messages(
        history=[],
        current_message="continue",
        harness_state="# Session Harness State\n\n## Recent Files\n- src/main.py",
    )

    system_content = messages[0]["content"]
    assert "# Session Harness State" in system_content
    assert "- src/main.py" in system_content


def test_history_is_trimmed_by_context_budget(tmp_path, monkeypatch) -> None:
    """历史消息过长时，只保留最近的合法片段。"""
    workspace = _make_workspace(tmp_path)
    monkeypatch.setattr(ContextBuilder, "MAX_HISTORY_CHARS", 60)
    builder = ContextBuilder(workspace)
    history = [
        {"role": "user", "content": "old question " + "x" * 80},
        {"role": "assistant", "content": "old answer " + "y" * 80},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]

    messages = builder.build_messages(history=history, current_message="now")
    contents = [message.get("content") for message in messages]

    assert all("old question" not in str(content) for content in contents)
    assert all("old answer" not in str(content) for content in contents)
    assert "recent question" in contents
    assert "recent answer" in contents


def test_history_budget_drops_orphan_tool_results(tmp_path, monkeypatch) -> None:
    """预算裁剪不能把孤立 tool 结果留给模型。"""
    workspace = _make_workspace(tmp_path)
    monkeypatch.setattr(ContextBuilder, "MAX_HISTORY_CHARS", 20)
    builder = ContextBuilder(workspace)
    history = [
        {"role": "user", "content": "old setup " + "x" * 50},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "x", "content": "ok"},
        {"role": "user", "content": "latest"},
    ]

    messages = builder.build_messages(history=history, current_message="now")

    assert all(message.get("role") != "tool" for message in messages)
    assert any(message.get("content") == "latest" for message in messages)


def test_runtime_context_is_separate_untrusted_user_message(tmp_path) -> None:
    """Runtime metadata should be merged with the user message."""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    messages = builder.build_messages(
        history=[],
        current_message="Return exactly: OK",
        channel="cli",
        chat_id="direct",
    )

    assert messages[0]["role"] == "system"
    assert "## Current Session" not in messages[0]["content"]

    # Runtime context is now merged with user message into a single message
    assert messages[-1]["role"] == "user"
    user_content = messages[-1]["content"]
    assert isinstance(user_content, str)
    assert ContextBuilder._RUNTIME_CONTEXT_TAG in user_content
    assert "Current Time:" in user_content
    assert "Channel: cli" in user_content
    assert "Chat ID: direct" in user_content
    assert "Return exactly: OK" in user_content
