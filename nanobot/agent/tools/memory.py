"""用户主动记忆工具：把明确要求记住的信息写入 USER.md。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class RememberUserInfoTool(Tool):
    """把用户明确要求长期保存的信息写入 USER.md。"""

    def __init__(self, workspace: Path):
        """记录工作区位置。"""
        self.workspace = workspace
        self.user_file = workspace / "USER.md"

    @property
    def name(self) -> str:
        """返回工具名称。"""
        return "remember_user_info"

    @property
    def description(self) -> str:
        """返回工具用途说明。"""
        return (
            "Update USER.md when the user explicitly asks you to remember personal "
            "facts, preferences, or profile information for future conversations. "
            "Read the current USER.md first if it exists, then pass the full updated "
            "markdown content. Do not use this for automatic conversation summaries, "
            "project status, implementation notes, tool output, or document contents."
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
            },
            "required": ["content"],
        }

    async def execute(self, content: str, reason: str | None = None) -> str:
        """写入完整的 USER.md 内容。"""
        if not content.strip():
            return "Error: USER.md content is empty; memory was not updated."

        self.user_file.write_text(content.rstrip() + "\n", encoding="utf-8")
        if reason and reason.strip():
            return f"USER.md updated. Reason: {reason.strip()}"
        return "USER.md updated."
