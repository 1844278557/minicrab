"""受限记忆 Agent：从新对话中维护 USER.md 和 MEMORY.md。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from minicrab.agent.memory_ops import MemoryOperationApplicator
from minicrab.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from minicrab.providers.base import LLMProvider
    from minicrab.session.manager import Session, SessionManager


class MemoryAgent:
    """在每轮回复后维护长期记忆，只允许写 USER.md 和 MEMORY.md。"""

    CURSOR_KEY = "memory_extracted_until"

    def __init__(self, workspace: Path, provider: LLMProvider, model: str):
        """保存工作区、模型提供者和允许写入的记忆文件。"""
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.user_file = workspace / "USER.md"
        self.memory_file = ensure_dir(workspace / "memory") / "MEMORY.md"
        self.memory_ops = MemoryOperationApplicator(workspace)

    async def extract_new(
        self,
        session: Session,
        sessions: SessionManager | None = None,
        *,
        start_index: int | None = None,
    ) -> bool:
        """扫描新增消息，必要时更新长期记忆，并推进扫描游标。"""
        end = len(session.messages)
        start = self._start_index(session, end, start_index)
        if start >= end:
            self._advance_cursor(session, end, sessions)
            return False

        new_messages = session.messages[start:end]
        if self._has_explicit_memory_tool_call(new_messages):
            logger.info("Memory extraction skipped for {}: explicit memory tool already used", session.key)
            self._advance_cursor(session, end, sessions)
            return False

        chunk = self._format_messages(new_messages)
        if not chunk.strip():
            logger.info("Memory extraction skipped for {}: no user/assistant text in new messages", session.key)
            self._advance_cursor(session, end, sessions)
            return False

        try:
            response = await self.provider.chat_with_retry(
                messages=self._build_prompt(chunk),
                tools=[],
                model=self.model,
                max_tokens=1800,
                temperature=0,
                tool_choice="none",
                response_format={"type": "json_object"},
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
        today = datetime.now().date().isoformat()
        current_user = self.user_file.read_text(encoding="utf-8") if self.user_file.exists() else ""
        current_memory = self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""
        return [
            {
                "role": "system",
                "content": (
                    "你是受限记忆 Agent，只负责从对话中维护长期记忆，并返回严格 JSON。"
                    "不要调用工具，不要输出 Markdown，不要解释。"
                    "USER.md 只保存稳定用户身份、偏好、个人事实、关系、日期、"
                    "以及用户对助手行为的长期要求。MEMORY.md 只保存跨会话仍有价值的"
                    "业务规则、长期项目决策、踩坑经验和重要外部信息位置。"
                    "自动写 USER.md 必须非常保守：只有用户明确表达长期偏好、稳定事实、"
                    "重要日期、关系或长期行为要求时才写。不要根据一次性任务、临时语气、"
                    "情绪状态、推测身份、敏感信息或低置信度判断写 USER.md。"
                    "不要保存当前任务进度、临时聊天、原始工具输出或完整文档内容。"
                    "如果本轮没有长期价值，必须返回 should_update=false。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "根据当前 USER.md、当前 MEMORY.md 和新增对话，判断是否需要更新长期记忆。\n\n"
                    "返回格式必须是严格 JSON：\n"
                    '{"should_update": true|false, "operations": ['
                    '{"target": "user|memory", "op": "append|replace|delete", '
                    '"section": "Markdown section title or empty string", '
                    '"match": "replace/delete 时精确匹配的旧 bullet，append 时为空字符串", '
                    '"content": "append/replace 时的一条新 bullet，delete 时为空字符串", '
                    '"confidence": "high|medium|low", "reason": "一句话原因"}], '
                    '"reason": "一句话总体原因"}\n\n'
                    "规则：\n"
                    "- 可以同时更新 user 和 memory。\n"
                    "- 不更新时 operations 必须是空数组。\n"
                    "- 只输出 append、replace、delete，不要输出完整文件内容。\n"
                    "- content 必须是一条 Markdown bullet，并以 '- ' 开头。\n"
                    "- replace/delete 必须在 match 中填入现有文件里完全一致的旧 bullet。\n"
                    "- 找不到精确旧条目时，不要猜测替换或删除。\n"
                    "- 只有 confidence=high 的操作才会执行；中低置信度应不更新。\n"
                    "- 优先更新已有条目，避免重复。\n"
                    "- 用户明确删除或忘记旧记忆时输出 delete。\n"
                    "- 用户明确纠正旧记忆时输出 replace。\n"
                    "- 明确新项目决策覆盖旧项目决策时输出 replace。\n"
                    "- 临时需求或不确定冲突时不更新。\n\n"
                    "MEMORY.md 条目格式：\n"
                    f"- 当前日期是 {today}。\n"
                    "- 每条长期工作记忆必须是一条 Markdown bullet。\n"
                    "- Business Rules 下使用：- [rule][YYYY-MM-DD] 内容。\n"
                    "- Project Decisions 下使用：- [decision][YYYY-MM-DD] 内容。\n"
                    "- Lessons Learned 下使用：- [lesson][YYYY-MM-DD] 内容。\n"
                    "- Important External Information 下使用：- [info][YYYY-MM-DD] 内容。\n"
                    "- 如果旧条目仍然有效，只更新它的内容或日期，不新增重复条目。\n\n"
                    "## Current USER.md\n"
                    f"{current_user or '(empty)'}\n\n"
                    "## Current MEMORY.md\n"
                    f"{current_memory or '(empty)'}\n\n"
                    "## New Conversation Messages\n"
                    f"{formatted_messages}"
                ),
            },
        ]

    def _apply_response(self, content: str | None, session_key: str) -> bool:
        """解析提取结果；只有合法 JSON 且目标受限时才写入记忆文件。"""
        if not content:
            logger.info("Memory extraction skipped for {}: empty response", session_key)
            return False
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            reason = self._classify_non_json_response(content)
            logger.warning("Memory extraction skipped for {}: {}", session_key, reason)
            return False
        if not isinstance(payload, dict) or not payload.get("should_update"):
            reason = payload.get("reason") if isinstance(payload, dict) else ""
            suffix = f" ({reason})" if reason else ""
            logger.info("Memory extraction skipped for {}: no durable memory{}", session_key, suffix)
            return False
        operations = payload.get("operations")
        if not isinstance(operations, list):
            logger.warning("Memory extraction skipped for {}: invalid operations", session_key)
            return False
        changed = False
        for operation in operations:
            if self._apply_operation(operation, session_key):
                changed = True
        if not changed:
            logger.info("Memory extraction skipped for {}: no operation applied", session_key)
        return changed

    def _apply_operation(self, operation: Any, session_key: str) -> bool:
        """应用单个高置信度受限记忆操作。"""
        if not isinstance(operation, dict):
            logger.warning("Memory extraction skipped one operation for {}: invalid item", session_key)
            return False
        confidence = str(operation.get("confidence") or "").lower()
        if confidence != "high":
            logger.info("Memory extraction skipped one operation for {}: confidence {}", session_key, confidence or "missing")
            return False
        result = self.memory_ops.apply(
            target=str(operation.get("target") or ""),
            op=str(operation.get("op") or ""),
            section=str(operation.get("section") or ""),
            match=str(operation.get("match") or ""),
            content=str(operation.get("content") or ""),
        )
        if result.changed:
            logger.info("Memory extraction applied operation for {}: {}", session_key, result.message)
        else:
            logger.info("Memory extraction skipped operation for {}: {}", session_key, result.message)
        return result.changed

    @staticmethod
    def _classify_non_json_response(content: str | None) -> str:
        """区分非 JSON 响应的格式违规类型。"""
        text = str(content or "").strip()
        if "```" in text:
            return "non-strict JSON response wrapped in markdown code block"
        if "{" in text or "}" in text or "[" in text or "]" in text:
            return "invalid or non-strict JSON-like response"
        return "plain-text non-JSON response"

    @staticmethod
    def _has_explicit_memory_tool_call(messages: list[dict[str, Any]]) -> bool:
        """当前轮已经显式写过记忆时，跳过后台自动提取，避免重复。"""
        for message in messages:
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if isinstance(function, dict) and function.get("name") in {"remember_user_info", "manage_memory"}:
                    return True
                if tool_call.get("name") in {"remember_user_info", "manage_memory"}:
                    return True
        return False

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        """只格式化用户和助手文本，避免把原始工具结果送入长期记忆提取。"""
        lines: list[str] = []
        for message in messages:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = MemoryAgent._text_content(message.get("content"))
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


class MemoryExtractor(MemoryAgent):
    """兼容旧名称；实际实现已经是受限 MemoryAgent。"""
