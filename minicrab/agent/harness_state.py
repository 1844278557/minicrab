"""维护会话级框架状态，用于压缩后恢复关键上下文。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


RECENT_FILES_KEY = "recent_files"
ACTIVE_SKILLS_KEY = "active_skills"
MAX_RECENT_FILES = 3
MAX_ACTIVE_SKILLS = 3
MAX_TOOL_EVIDENCE_ITEMS = 20

ALWAYS_CONTEXT_FILES = {
    "AGENTS.md",
    "SOUL.md",
    "USER.md",
    "CONTEXT.md",
    "TOOLS.md",
    "memory/MEMORY.md",
}

TOOL_SKILL_MAP = {
    "manage_memory": "memory",
    "remember_user_info": "memory",
    "index_document": "rag",
    "search_rag": "rag",
    "list_rag_documents": "rag",
    "remove_rag_document": "rag",
    "clear_rag_store": "rag",
    "cleanup_expired_rag": "rag",
    "search_episodes": "episode",
    "list_recent_episodes": "episode",
    "get_episode_stats": "episode",
}

TOOL_PATH_ARGS = {
    "read_file": ("path",),
    "write_file": ("path",),
    "edit_file": ("path",),
    "index_document": ("file_path",),
}


def update_session_harness_state(session: Any, messages: list[dict[str, Any]], workspace: Path) -> bool:
    """从新保存的工具调用中更新最近文件和已激活 skill。"""
    recent_files = _coerce_list(session.metadata.get(RECENT_FILES_KEY))
    active_skills = _coerce_list(session.metadata.get(ACTIVE_SKILLS_KEY))
    before_files = list(recent_files)
    before_skills = list(active_skills)

    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            name, args = parse_tool_call(tool_call)
            if not name:
                continue

            skill = TOOL_SKILL_MAP.get(name)
            if skill:
                active_skills = _move_to_front(active_skills, skill, MAX_ACTIVE_SKILLS)

            for arg_name in TOOL_PATH_ARGS.get(name, ()):
                path = _normalize_recent_file(args.get(arg_name), workspace)
                if path:
                    recent_files = _move_to_front(recent_files, path, MAX_RECENT_FILES)

    session.metadata[RECENT_FILES_KEY] = recent_files
    session.metadata[ACTIVE_SKILLS_KEY] = active_skills
    return recent_files != before_files or active_skills != before_skills


def format_harness_state(metadata: dict[str, Any] | None) -> str:
    """把框架状态格式化为短上下文块。"""
    if not metadata:
        return ""

    recent_files = _coerce_list(metadata.get(RECENT_FILES_KEY))
    active_skills = _coerce_list(metadata.get(ACTIVE_SKILLS_KEY))
    if not recent_files and not active_skills:
        return ""

    parts = ["# Session Harness State"]
    if recent_files:
        parts.append("## Recent Files\n" + "\n".join(f"- {path}" for path in recent_files))
    if active_skills:
        parts.append("## Active Skills\n" + "\n".join(f"- {skill}" for skill in active_skills))
    return "\n\n".join(parts)


def build_tool_evidence(messages: list[dict[str, Any]], workspace: Path) -> str:
    """从待压缩消息中确定性提取工具证据，避免只依赖 LLM 记忆工具结果。"""
    tool_results = _tool_results_by_id(messages)
    items: list[str] = []

    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            name, args = parse_tool_call(tool_call)
            if not name:
                continue
            result = tool_results.get(str(tool_call.get("id") or ""))
            item = _format_tool_evidence_item(name, args, result, workspace)
            if item:
                items.append(item)
            if len(items) >= MAX_TOOL_EVIDENCE_ITEMS:
                break
        if len(items) >= MAX_TOOL_EVIDENCE_ITEMS:
            break

    if not items:
        return ""
    return "## Tool Evidence\n" + "\n".join(items)


def parse_tool_call(tool_call: Any) -> tuple[str, dict[str, Any]]:
    """解析 OpenAI 风格或简化风格的工具调用。"""
    if not isinstance(tool_call, dict):
        return "", {}

    function = tool_call.get("function")
    if isinstance(function, dict):
        name = str(function.get("name") or "")
        raw_args = function.get("arguments")
    else:
        name = str(tool_call.get("name") or "")
        raw_args = tool_call.get("arguments")

    args = _parse_args(raw_args)
    return name, args


def _parse_args(raw_args: Any) -> dict[str, Any]:
    """解析工具参数。"""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_results_by_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    """按 tool_call_id 收集工具结果。"""
    results: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            results[str(tool_call_id)] = _content_to_text(message.get("content"))
    return results


def _format_tool_evidence_item(
    name: str,
    args: dict[str, Any],
    result: str | None,
    workspace: Path,
) -> str:
    """格式化一条工具证据。"""
    fields: list[str] = []
    for key in ("path", "file_path", "query", "command", "target", "op", "action"):
        if key in args and args[key] not in (None, ""):
            value = args[key]
            if key in {"path", "file_path"}:
                value = _normalize_recent_file(value, workspace) or str(value)
            fields.append(f"{key}={_shorten(str(value), 120)}")

    detail = f" ({', '.join(fields)})" if fields else ""
    summary = _summarize_tool_result(result or "")
    if summary:
        return f"- {name}{detail}: {summary}"
    return f"- {name}{detail}"


def _summarize_tool_result(content: str) -> str:
    """用规则提取工具结果中最有用的一行。"""
    text = content.strip()
    if not text:
        return "no visible result"

    patterns = (
        r"\d+\s+passed(?:,\s*\d+\s+skipped)?",
        r"\d+\s+failed(?:,\s*\d+\s+passed)?",
        r"FAILED\s+.*",
        r"Error:\s*.*",
        r"Traceback\s+\(most recent call last\):",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _shorten(match.group(0).strip(), 180)

    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return _shorten(stripped, 180)
    return "no visible result"


def _normalize_recent_file(value: Any, workspace: Path) -> str | None:
    """把文件路径转成工作区相对路径，并排除固定注入文件。"""
    if not isinstance(value, str) or not value.strip():
        return None

    raw = Path(value.strip())
    try:
        workspace_resolved = workspace.expanduser().resolve()
        resolved = raw.expanduser().resolve() if raw.is_absolute() else (workspace_resolved / raw).resolve()
        relative = resolved.relative_to(workspace_resolved)
    except (OSError, ValueError):
        return None

    normalized = relative.as_posix()
    if normalized in ALWAYS_CONTEXT_FILES:
        return None
    return normalized


def _move_to_front(items: list[str], item: str, limit: int) -> list[str]:
    """把条目移动到列表最前面并限制长度。"""
    cleaned = [existing for existing in items if existing != item]
    return [item, *cleaned][:limit]


def _coerce_list(value: Any) -> list[str]:
    """把元数据值整理成字符串列表。"""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _content_to_text(content: Any) -> str:
    """把工具结果内容转成文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _shorten(text: str, limit: int) -> str:
    """限制单行摘要长度。"""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
