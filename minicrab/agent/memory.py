"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import inspect
import json
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from minicrab.agent.episode import EpisodeStore
from minicrab.agent.harness_state import build_tool_evidence, format_harness_state
from minicrab.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from minicrab.providers.base import LLMProvider
    from minicrab.session.manager import Session, SessionManager


_SAVE_SESSION_SUMMARY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_session_summary",
            "description": "Save the updated current-session summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Brief topic/title of this conversation segment (5-10 words).",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Full updated session summary in markdown. Include prior summary details "
                        "that are still needed for the current conversation plus important new context.",
                    },
                },
                "required": ["summary"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_tool_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MemoryStore:
    """
    Memory-backed storage helpers.

    MEMORY.md is long-term working memory. Session compaction returns a
    running summary to the caller instead of writing MEMORY.md or Episodes.
    """

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.history_file = self.memory_dir / "HISTORY.md"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.episode_store = EpisodeStore(self.memory_dir)
        self._consecutive_failures = 0

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, content: Any) -> None:
        """为旧集成追加一条历史记录。"""
        text = _ensure_text(content).strip()
        if not text:
            return
        previous = self.history_file.read_text(encoding="utf-8") if self.history_file.exists() else ""
        separator = "\n" if previous and not previous.endswith("\n") else ""
        self.history_file.write_text(f"{previous}{separator}{text}\n", encoding="utf-8")

    def get_memory_context(self, max_chars: int | None = None) -> str:
        long_term = self.read_long_term()
        if max_chars and max_chars > 0 and len(long_term) > max_chars:
            long_term = (
                "[Long-term memory truncated to fit context budget]\n"
                + long_term[-max_chars:]
            )
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
        session_id: str = "unknown",
        current_summary: str = "",
        tool_evidence: str = "",
        harness_state: str = "",
    ) -> str | None:
        """Compact an old message chunk into an updated current-session summary."""
        if not messages:
            return current_summary

        prompt = f"""Update the current-session summary and call save_session_summary.

This summary is only for continuing the current session after old raw messages
are removed from the prompt. Do not write long-term memory here.

Return the summary as markdown with exactly these headings:
- # Current Session Summary
- ## Primary Request
- ## Current State
- ## Key Decisions
- ## Important Files / Data
- ## Tool Results That Matter
- ## Errors And Fixes
- ## Pending Tasks
- ## Recent User Requirements
- ## Rehydration Anchors
- ## Next Step

Use Tool Evidence for tool results. Do not copy raw tool output. Keep concrete
file paths, decisions, errors, and pending work that are needed to continue.
Preserve rehydration anchors from Harness State when they are still relevant.

## Existing Current-Session Summary
{current_summary or "(empty)"}

## Harness State
{harness_state or "(empty)"}

## Tool Evidence
{tool_evidence or "(empty)"}

## Conversation to Process
{self._format_messages(messages)}"""

        chat_messages = [
            {"role": "system", "content": "You are a session compaction agent. Return an updated markdown summary for the current conversation only. Use the required headings. Preserve current goal, decisions, unresolved issues, important files, active skills, tool evidence, and facts needed to continue. Do not create long-term memory, do not archive episodes, and do not include raw tool output."},
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_session_summary"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_SESSION_SUMMARY_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(
                response.content
            ):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_SESSION_SUMMARY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Session compaction: LLM did not call save_session_summary "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_summary(messages, current_summary, session_id)

            args = _normalize_tool_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Session compaction: unexpected save_session_summary arguments")
                return self._fail_or_raw_summary(messages, current_summary, session_id)

            summary_source = args.get("summary")
            if summary_source is None:
                logger.warning("Session compaction: summary is missing")
                return self._fail_or_raw_summary(messages, current_summary, session_id)

            summary = _ensure_text(summary_source).strip()
            if not summary:
                logger.warning("Session compaction: summary is empty")
                return self._fail_or_raw_summary(messages, current_summary, session_id)
            self._consecutive_failures = 0
            logger.info(
                "Session compaction done for {}: {} messages summarized",
                session_id,
                len(messages),
            )
            return summary
        except Exception:
            logger.exception("Session compaction failed")
            return self._fail_or_raw_summary(messages, current_summary, session_id)

    def _fail_or_raw_summary(
        self,
        messages: list[dict],
        current_summary: str = "",
        session_id: str = "unknown",
    ) -> str | None:
        """失败计数；连续失败后返回确定性兜底摘要，不写长期文件。"""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return None
        fallback = self._raw_summary(messages, current_summary, session_id)
        self._consecutive_failures = 0
        return fallback

    def _raw_summary(
        self,
        messages: list[dict],
        current_summary: str = "",
        session_id: str = "unknown",
    ) -> str:
        """确定性兜底摘要，避免压缩流程永久卡住。"""
        formatted = self._format_messages(messages)
        if len(formatted) > 4000:
            formatted = formatted[:2000] + "\n...\n" + formatted[-2000:]
        raw_lines = [
            current_summary.strip(),
            f"## Compacted Session Chunk ({session_id})",
            f"{len(messages)} messages were compacted without LLM summarization.",
            formatted,
        ]
        logger.warning(
            "Session compaction degraded: raw-summary fallback for {} messages in {}",
            len(messages),
            session_id,
        )
        return "\n\n".join(line for line in raw_lines if line)


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5
    SESSION_SUMMARY_KEY = "running_summary"

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift
    _TRIGGER_RATIO = 0.8
    _RECENT_USER_TURNS_TO_KEEP = 10

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
    ):
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(
        self,
        messages: list[dict[str, object]],
        session_id: str = "unknown",
        current_summary: str = "",
        tool_evidence: str = "",
        harness_state: str = "",
    ) -> str | None:
        """Compact a selected message chunk into a session summary."""
        return await self.store.consolidate(
            messages,
            self.provider,
            self.model,
            session_id,
            current_summary=current_summary,
            tool_evidence=tool_evidence,
            harness_state=harness_state,
        )

    async def _call_consolidate_messages(
        self,
        messages: list[dict[str, object]],
        session_id: str,
        current_summary: str = "",
        tool_evidence: str = "",
        harness_state: str = "",
    ) -> Any:
        """调用记忆整理，同时兼容旧的一参数替身函数。"""
        try:
            parameters = inspect.signature(self.consolidate_messages).parameters
        except (TypeError, ValueError):
            parameters = {}
        if len(parameters) <= 1:
            return await self.consolidate_messages(messages)
        if len(parameters) <= 2:
            return await self.consolidate_messages(messages, session_id)
        if len(parameters) <= 3:
            return await self.consolidate_messages(messages, session_id, current_summary)
        if len(parameters) <= 4:
            return await self.consolidate_messages(
                messages, session_id, current_summary, tool_evidence
            )
        return await self.consolidate_messages(
            messages, session_id, current_summary, tool_evidence, harness_state
        )

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        preserve_from = self._recent_user_turn_start(session)
        if start >= preserve_from:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        end = min(preserve_from + 1, len(session.messages))
        for idx in range(start, end):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def _recent_user_turn_start(self, session: Session) -> int:
        """返回需要保留的最近用户轮次起点。"""
        keep = max(0, int(self._RECENT_USER_TURNS_TO_KEEP))
        if keep <= 0:
            return len(session.messages)

        seen = 0
        for idx in range(len(session.messages) - 1, session.last_consolidated - 1, -1):
            if session.messages[idx].get("role") == "user":
                seen += 1
                if seen >= keep:
                    return idx
        return session.last_consolidated

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
            session_summary=session.metadata.get(self.SESSION_SUMMARY_KEY),
            harness_state=format_harness_state(session.metadata),
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(
        self,
        messages: list[dict[str, object]],
        session_id: str = "unknown",
    ) -> bool:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback)."""
        if not messages:
            return True
        for _ in range(self.store._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            if await self._call_consolidate_messages(messages, session_id):
                return True
        return True

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            trigger = max(1, int(budget * self._TRIGGER_RATIO))
            target = budget // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < trigger:
                logger.debug(
                    "Token consolidation idle {}: {}/{} trigger={} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    trigger,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                summary_result = await self._call_consolidate_messages(
                    chunk,
                    session.key,
                    str(session.metadata.get(self.SESSION_SUMMARY_KEY) or ""),
                    build_tool_evidence(chunk, self.sessions.workspace),
                    format_harness_state(session.metadata),
                )
                if not summary_result:
                    return
                if isinstance(summary_result, str):
                    session.metadata[self.SESSION_SUMMARY_KEY] = summary_result
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
