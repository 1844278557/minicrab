"""用户主动记忆工具：把明确要求记住的信息写入长期记忆文件。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minicrab.agent.memory_ops import MemoryOperationApplicator
from minicrab.agent.tools.base import Tool


class ManageMemoryTool(Tool):
    """按单条操作维护 USER.md 或 MEMORY.md。"""

    def __init__(self, workspace: Path):
        """记录工作区位置和受限操作器。"""
        self.applicator = MemoryOperationApplicator(workspace)

    @property
    def name(self) -> str:
        """返回工具名称。"""
        return "manage_memory"

    @property
    def description(self) -> str:
        """返回工具用途说明。"""
        return (
            "Append, replace, or delete exactly one long-term memory bullet when the "
            "user explicitly asks to remember, correct, or forget something. Use "
            "target='user' for stable user identity, preferences, personal facts, "
            "relationships, dates, or long-term assistant behavior requirements. Use "
            "target='memory' for business rules, durable project decisions, lessons "
            "learned, or important external information locations. For replace/delete, "
            "read the target memory file first and pass the exact existing bullet in "
            "match. For target='memory', content must use [rule|decision|lesson|info]"
            "[YYYY-MM-DD] under the matching section. Never use this for temporary task "
            "progress, raw tool output, full document contents, or inferred low-confidence "
            "personal facts."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """返回工具参数定义。"""
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["user", "memory"],
                    "description": "Target memory file: 'user' for USER.md, 'memory' for memory/MEMORY.md.",
                },
                "op": {
                    "type": "string",
                    "enum": ["append", "replace", "delete"],
                    "description": "Memory operation to apply.",
                },
                "section": {
                    "type": ["string", "null"],
                    "description": "Markdown section title for append, e.g. Preferences or Project Decisions.",
                },
                "match": {
                    "type": ["string", "null"],
                    "description": "Exact existing bullet for replace/delete.",
                },
                "content": {
                    "type": ["string", "null"],
                    "description": "Single replacement or appended memory bullet. Must start with '- '.",
                },
                "reason": {
                    "type": ["string", "null"],
                    "description": "Optional short reason for the memory change.",
                },
            },
            "required": ["target", "op"],
        }

    async def execute(
        self,
        target: str,
        op: str,
        section: str | None = None,
        match: str | None = None,
        content: str | None = None,
        reason: str | None = None,
    ) -> str:
        """执行单条记忆操作。"""
        result = self.applicator.apply(
            target=target,
            op=op,
            section=section,
            match=match,
            content=content,
        )
        if reason and reason.strip():
            return f"{result.message} Reason: {reason.strip()}"
        return result.message


class RememberUserInfoTool(Tool):
    """把用户明确要求长期保存的信息写入 USER.md 或 MEMORY.md。"""

    def __init__(self, workspace: Path):
        """记录工作区位置。"""
        self.workspace = workspace
        self.user_file = workspace / "USER.md"
        self.memory_file = workspace / "memory" / "MEMORY.md"

    @property
    def name(self) -> str:
        """返回工具名称。"""
        return "remember_user_info"

    @property
    def description(self) -> str:
        """返回工具用途说明。"""
        return (
            "Legacy full-file memory update kept for compatibility. Prefer manage_memory "
            "for new remember/correct/forget requests because it applies one safe memory "
            "operation at a time. Use target='user' for stable user "
            "identity, personal facts, preferences, relationships, dates, or assistant "
            "behavior requirements. Use target='memory' for durable cross-session "
            "experience, business rules, long-term project decisions, or important "
            "external information locations. For target='memory', keep one memory per "
            "bullet and use [rule|decision|lesson|info][YYYY-MM-DD] prefixes under the "
            "matching MEMORY.md section. Read the current target file first if it "
            "exists, then pass the full updated markdown content. Do not use this for "
            "automatic conversation summaries, temporary task progress, raw tool output, "
            "or full document contents."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """返回工具参数定义。"""
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Full updated USER.md content in markdown.",
                    "minLength": 1,
                },
                "reason": {
                    "type": ["string", "null"],
                    "description": "Optional short reason for this memory update.",
                },
                "target": {
                    "type": "string",
                    "enum": ["user", "memory"],
                    "default": "user",
                    "description": "Write target: 'user' for USER.md, 'memory' for memory/MEMORY.md.",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        reason: str | None = None,
        target: str = "user",
    ) -> str:
        """按目标写入完整的长期记忆内容。"""
        if not content.strip():
            return "Error: memory content is empty; memory was not updated."

        normalized_target = target or "user"
        if normalized_target not in {"user", "memory"}:
            return "Error: target must be 'user' or 'memory'; memory was not updated."

        target_file = self.user_file if normalized_target == "user" else self.memory_file
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(content.rstrip() + "\n", encoding="utf-8")
        filename = "USER.md" if normalized_target == "user" else "MEMORY.md"
        if reason and reason.strip():
            return f"{filename} updated. Reason: {reason.strip()}"
        return f"{filename} updated."
