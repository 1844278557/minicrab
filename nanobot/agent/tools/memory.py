"""用户主动记忆工具：把明确要求记住的信息写入长期记忆文件。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


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
            "Update long-term memory only when the user explicitly asks you to remember "
            "information for future conversations. Use target='user' for stable user "
            "identity, personal facts, preferences, relationships, dates, or assistant "
            "behavior requirements. Use target='memory' for durable cross-session "
            "experience, business rules, long-term project decisions, or important "
            "external information locations. Read the current target file first if it "
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
