"""测试会话框架状态和工具证据提取。"""

from __future__ import annotations

import json
from pathlib import Path

from minicrab.agent.harness_state import (
    ACTIVE_SKILLS_KEY,
    RECENT_FILES_KEY,
    build_tool_evidence,
    format_harness_state,
    update_session_harness_state,
)
from minicrab.session.manager import Session


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    """创建 OpenAI 风格工具调用。"""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def test_harness_state_tracks_recent_files_and_skips_always_loaded(tmp_path: Path) -> None:
    """最近文件应只记录动态访问的工作区文件。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = Session(key="web:default")

    changed = update_session_harness_state(
        session,
        [
            {
                "role": "assistant",
                "tool_calls": [
                    _tool_call("1", "read_file", {"path": str(workspace / "USER.md")}),
                    _tool_call("2", "read_file", {"path": str(workspace / "docs" / "a.md")}),
                    _tool_call("3", "edit_file", {"path": "src/main.py"}),
                    _tool_call("4", "index_document", {"file_path": "docs/report.pdf"}),
                    _tool_call("5", "write_file", {"path": "notes/todo.md"}),
                ],
            }
        ],
        workspace,
    )

    assert changed is True
    assert session.metadata[RECENT_FILES_KEY] == [
        "notes/todo.md",
        "docs/report.pdf",
        "src/main.py",
    ]


def test_harness_state_tracks_active_skills_as_mru(tmp_path: Path) -> None:
    """已激活 skill 应按最近使用顺序记录。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = Session(key="web:default")

    update_session_harness_state(
        session,
        [
            {
                "role": "assistant",
                "tool_calls": [
                    _tool_call("1", "manage_memory", {"target": "user", "op": "append"}),
                    _tool_call("2", "search_rag", {"query": "合同"}),
                    _tool_call("3", "search_episodes", {"query": "旧对话"}),
                    _tool_call("4", "remember_user_info", {"target": "memory"}),
                ],
            }
        ],
        workspace,
    )

    assert session.metadata[ACTIVE_SKILLS_KEY] == ["memory", "episode", "rag"]


def test_format_harness_state_returns_short_markdown() -> None:
    """框架状态应能格式化为短 Markdown 块。"""
    text = format_harness_state({
        RECENT_FILES_KEY: ["src/main.py"],
        ACTIVE_SKILLS_KEY: ["memory"],
    })

    assert "# Session Harness State" in text
    assert "- src/main.py" in text
    assert "- memory" in text


def test_tool_evidence_extracts_tool_calls_and_results(tmp_path: Path) -> None:
    """工具证据应包含关键参数和规则提取的结果摘要。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                _tool_call("call_1", "read_file", {"path": str(workspace / "src" / "main.py")}),
                _tool_call("call_2", "exec", {"command": ".venv\\Scripts\\python.exe -m pytest"}),
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "print('ok')"},
        {"role": "tool", "tool_call_id": "call_2", "name": "exec", "content": "12 passed, 1 skipped"},
    ]

    evidence = build_tool_evidence(messages, workspace)

    assert "read_file (path=src/main.py): print('ok')" in evidence
    assert "exec (command=.venv\\Scripts\\python.exe -m pytest): 12 passed, 1 skipped" in evidence
