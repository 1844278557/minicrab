"""自动工作记忆提取：从新对话中提取可长期复用的 MEMORY.md 信息。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


class MemoryExtractor:
    """在每轮回复后提取长期工作记忆，不处理 USER.md。"""

    CURSOR_KEY = "memory_extracted_until"

    def __init__(self, workspace: Path, provider: LLMProvider, model: str):
        """保存工作区、模型提供者和目标记忆文件。"""
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.memory_file = ensure_dir(workspace / "memory") / "MEMORY.md"

    async def extract_new(
        self,
        session: Session,
        sessions: SessionManager | None = None,
        *,
        start_index: int | None = None,
    ) -> bool:
        """扫描新增消息，必要时更新 MEMORY.md，并推进扫描游标。"""
        end = len(session.messages)
        start = self._start_index(session, end, start_index)
        if start >= end:
            self._advance_cursor(session, end, sessions)
            return False

        chunk = self._format_messages(session.messages[start:end])
        if not chunk.strip():
            self._advance_cursor(session, end, sessions)
            return False

        try:
            response = await self.provider.chat_with_retry(
                messages=self._build_prompt(chunk),
                tools=[],
                model=self.model,
                max_tokens=900,
                temperature=0,
                tool_choice="none",
            )
        except Exception:
            logger.exception("Memory extraction failed")
            self._advance_cursor(session, end, sessions)
            return False

        updated = self._apply_response(response.content, session.key)
        self._advance_cursor(session, end, sessions)
        return updated

    def _start_index(
        self,
        session: Session,
        end: int,
        start_index: int | None,
    ) -> int:
        """决定本次从哪里开始扫描；无游标时只扫当前轮。"""
        raw_cursor = session.metadata.get(self.CURSOR_KEY)
        if raw_cursor is None:
            start = 0 if start_index is None else start_index
        else:
            try:
                start = int(raw_cursor)
            except (TypeError, ValueError):
                start = 0 if start_index is None else start_index
        if start > end:
            return 0
        return max(0, start)

    def _advance_cursor(
        self,
        session: Session,
        end: int,
        sessions: SessionManager | None,
    ) -> None:
        """记录已扫描到的消息位置。"""
        session.metadata[self.CURSOR_KEY] = end
        if sessions is not None:
            sessions.save(session)

    def _build_prompt(self, formatted_messages: str) -> list[dict[str, str]]:
        """构造严格 JSON 输出的提取提示。"""
        current_memory = self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""
        return [
            {
                "role": "system",
                "content": (
                    "你只负责从对话中提取长期工作记忆，并返回严格 JSON。"
                    "不要调用工具，不要输出 Markdown，不要解释。"
                    "只保存跨会话仍有价值的经验、业务规则、长期项目决策、"
                    "重要外部信息位置。不要保存用户身份、偏好、个人事实、"
                    "当前任务进度、临时聊天、原始工具输出或完整文档内容。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "根据当前 MEMORY.md 和新增对话，判断是否需要更新 MEMORY.md。\n\n"
                    "返回格式必须是严格 JSON：\n"
                    '{"should_update": true|false, "memory_update": "完整更新后的 MEMORY.md，'
                    '不更新时为空字符串", "reason": "一句话原因"}\n\n'
                    "## Current MEMORY.md\n"
                    f"{current_memory or '(empty)'}\n\n"
                    "## New Conversation Messages\n"
                    f"{formatted_messages}"
                ),
            },
        ]

    def _apply_response(self, content: str | None, session_key: str) -> bool:
        """解析提取结果；只有合法 JSON 且明确更新时才写 MEMORY.md。"""
        if not content:
            return False
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            logger.warning("Memory extraction skipped for {}: non-JSON response", session_key)
            return False
        if not isinstance(payload, dict) or not payload.get("should_update"):
            return False
        memory_update = payload.get("memory_update")
        if not isinstance(memory_update, str) or not memory_update.strip():
            logger.warning("Memory extraction skipped for {}: empty memory_update", session_key)
            return False
        previous = self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""
        normalized = memory_update.rstrip() + "\n"
        if normalized == previous:
            return False
        self.memory_file.write_text(normalized, encoding="utf-8")
        logger.info("Memory extraction updated MEMORY.md for {}", session_key)
        return True

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        """只格式化用户和助手文本，避免把原始工具结果送入长期记忆提取。"""
        lines: list[str] = []
        for message in messages:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = MemoryExtractor._text_content(message.get("content"))
            if not content:
                continue
            lines.append(f"{str(role).upper()}: {content}")
        return "\n\n".join(lines)

    @staticmethod
    def _text_content(content: Any) -> str:
        """把 session 内容转成适合记忆提取的文本。"""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"].strip())
            return "\n".join(part for part in parts if part)
        return ""
