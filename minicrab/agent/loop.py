"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import AsyncExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from minicrab.agent.approval import ApprovalManager
from minicrab.agent.context import ContextBuilder
from minicrab.agent.episode import EpisodeStore
from minicrab.agent.harness_state import format_harness_state, update_session_harness_state
from minicrab.config.defaults import get_default_data_dir
from minicrab.agent.hook import AgentHook, AgentHookContext
from minicrab.agent.memory import MemoryConsolidator
from minicrab.agent.memory_extractor import MemoryAgent
from minicrab.agent.rag.store import RAGStore
from minicrab.agent.runner import AgentRunner, AgentRunSpec
from minicrab.agent.skills import BUILTIN_SKILLS_DIR
from minicrab.agent.subagent import SubagentManager
from minicrab.agent.tools.cron import CronTool
from minicrab.agent.tools.episode import (
    GetEpisodeStatsTool,
    ListRecentEpisodesTool,
    SearchEpisodesTool,
)
from minicrab.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from minicrab.agent.tools.memory import ManageMemoryTool, RememberUserInfoTool
from minicrab.agent.tools.message import MessageTool
from minicrab.agent.tools.rag import (
    CleanupExpiredRAGTool,
    ClearRAGStoreTool,
    IndexDocumentTool,
    ListRAGDocumentsTool,
    RemoveRAGDocumentTool,
    SearchRAGTool,
)
from minicrab.agent.tools.registry import ToolRegistry
from minicrab.agent.tools.shell import ExecTool
from minicrab.agent.tools.spawn import SpawnTool
from minicrab.agent.tools.web import WebFetchTool, WebSearchTool
from minicrab.bus.events import InboundMessage, OutboundMessage
from minicrab.bus.queue import MessageBus
from minicrab.command import CommandContext, CommandRouter, register_builtin_commands
from minicrab.providers.base import LLMProvider, ToolCallRequest
from minicrab.session.manager import Session, SessionManager
from minicrab.utils.helpers import build_assistant_message, is_approval_prompt_text

if TYPE_CHECKING:
    from minicrab.config.schema import (
        ChannelsConfig,
        ExecToolConfig,
        RAGConfig,
        WebSearchConfig,
    )
    from minicrab.cron.service import CronService


@dataclass(slots=True)
class AgentLoopRunResult:
    """AgentLoop 内部执行结果，兼容旧的三元组解包。"""

    final_content: str | None
    tools_used: list[str]
    messages: list[dict[str, Any]]
    approval_event: dict[str, Any] | None = None
    stop_reason: str = "completed"

    def __iter__(self):
        """保持原有 final_content, tools_used, messages 解包方式可用。"""
        yield self.final_content
        yield self.tools_used
        yield self.messages


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000
    _INDEXABLE_UPLOAD_SUFFIXES = {
        ".pdf", ".docx", ".doc", ".md", ".txt", ".csv", ".json", ".html", ".htm", ".py"
    }
    _SHORT_AFFIRMATIVES = {
        "是", "是的", "确认", "可以", "可以的", "好", "好的", "行", "开始", "同意", "嗯", "对"
    }

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        rag_config: RAGConfig | None = None,
        memory_provider: LLMProvider | None = None,
        memory_model: str | None = None,
    ):
        from minicrab.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}

        self.context = ContextBuilder(workspace, timezone=timezone)
        self.approvals = ApprovalManager(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)
        self.episode_store = EpisodeStore(workspace / "memory")
        self.rag_store = RAGStore(
            workspace / "memory",
            ttl_days=rag_config.ttl_days if rag_config else 30,
            embedding_model=rag_config.embedding_model if rag_config else "all-MiniLM-L6-v2",
            embedding_api_key=rag_config.siliconflow_api_key if rag_config else None,
            embedding_base_url=rag_config.siliconflow_embedding_base_url if rag_config else None,
            chunk_size=rag_config.chunk_size if rag_config else 800,
            chunk_overlap=rag_config.chunk_overlap if rag_config else 100,
        )
        self.rag_config = rag_config
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # 新环境变量优先，旧前缀继续兼容；小于等于 0 表示不限并发。
        _max = int(os.environ.get("MINICRAB_MAX_CONCURRENT_REQUESTS", os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3")))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
        )
        self.memory_agent = MemoryAgent(
            workspace,
            memory_provider or provider,
            memory_model or self.model,
        )
        self.memory_extractor = self.memory_agent
        self._register_default_tools()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        self.tools.register(ManageMemoryTool(workspace=self.workspace))
        self.tools.register(RememberUserInfoTool(workspace=self.workspace))
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
            ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )
        self.tools.register(SearchEpisodesTool(self.episode_store))
        self.tools.register(ListRecentEpisodesTool(self.episode_store))
        self.tools.register(GetEpisodeStatsTool(self.episode_store))
        self.tools.register(IndexDocumentTool(store=self.rag_store))

        rerank_key = None
        rerank_model = "Qwen/Qwen3-Reranker-8B"
        rerank_base_url = "https://api.siliconflow.cn/v1"

        if self.rag_config:
            rerank_key = self.rag_config.siliconflow_api_key
            rerank_model = self.rag_config.rerank_model
            rerank_base_url = self.rag_config.rerank_base_url

        if not rerank_key:
            try:
                sf_cfg = getattr(self.provider.config, "siliconflow", None)
                if sf_cfg and hasattr(sf_cfg, "api_key") and sf_cfg.api_key:
                    rerank_key = sf_cfg.api_key
                    logger.debug("RAG rerank: using siliconflow provider api_key")
            except Exception:
                pass

        if not rerank_key:
            logger.info(
                "RAG reranking disabled: no API key configured. "
                "Set rag.siliconFlowApiKey in config or SILICONFLOW_API_KEY env var to enable."
            )

        self.tools.register(SearchRAGTool(
            store=self.rag_store,
            rerank_api_key=rerank_key,
            rerank_model=rerank_model,
            rerank_base_url=rerank_base_url,
        ))
        self.tools.register(RemoveRAGDocumentTool(store=self.rag_store))
        self.tools.register(ListRAGDocumentsTool(store=self.rag_store))
        self.tools.register(ClearRAGStoreTool(store=self.rag_store))
        self.tools.register(CleanupExpiredRAGTool(store=self.rag_store))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from minicrab.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(
        self,
        channel: str,
        chat_id: str,
        message_id: str | None = None,
        *,
        sender_id: str = "user",
        session_key: str | None = None,
    ) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from minicrab.utils.helpers import strip_think
        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        session_key: str | None = None,
        tools_enabled: bool = True,
    ) -> AgentLoopRunResult:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.
        """
        loop_self = self

        class _LoopHook(AgentHook):
            def __init__(self) -> None:
                self._stream_buf = ""

            def wants_streaming(self) -> bool:
                return on_stream is not None

            async def on_stream(self, context: AgentHookContext, delta: str) -> None:
                from minicrab.utils.helpers import strip_think

                prev_clean = strip_think(self._stream_buf)
                self._stream_buf += delta
                new_clean = strip_think(self._stream_buf)
                incremental = new_clean[len(prev_clean):]
                if incremental and on_stream:
                    await on_stream(incremental)

            async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
                if on_stream_end:
                    await on_stream_end(resuming=resuming)
                self._stream_buf = ""

            async def before_execute_tools(self, context: AgentHookContext) -> None:
                if on_progress:
                    if not on_stream:
                        thought = loop_self._strip_think(context.response.content if context.response else None)
                        if thought:
                            await on_progress(thought)
                    tool_hint = loop_self._strip_think(loop_self._tool_hint(context.tool_calls))
                    await on_progress(tool_hint, tool_hint=True)
                for tc in context.tool_calls:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tc.name, args_str[:200])
                loop_self._set_tool_context(channel, chat_id, message_id)

            def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
                return loop_self._strip_think(content)

        def _request_approval(
            tool_call: ToolCallRequest,
            tool_calls: list[ToolCallRequest],
            messages: list[dict[str, Any]],
            response: Any,
        ) -> dict[str, Any] | None:
            arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
            return self.approvals.request(
                session_key=session_key or f"{channel}:{chat_id}",
                channel=channel,
                chat_id=chat_id,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                arguments=arguments,
                resume_messages=messages,
                tool_calls=tool_calls,
                response_content=response.content,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
                resume_save_skip=len(messages),
            )

        result = await self.runner.run(AgentRunSpec(
            initial_messages=initial_messages,
            tools=self.tools,
            model=self.model,
            max_iterations=self.max_iterations,
            hook=_LoopHook(),
            error_message="Sorry, I encountered an error calling the AI model.",
            concurrent_tools=True,
            approval_callback=_request_approval,
            tools_enabled=tools_enabled,
            stop_after_tools_callback=lambda: (
                (mt := self.tools.get("message")) is not None
                and isinstance(mt, MessageTool)
                and mt._sent_in_turn
            ),
        ))
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return AgentLoopRunResult(
            final_content=result.final_content,
            tools_used=result.tools_used,
            messages=result.messages,
            approval_event=result.approval_event,
            stop_reason=result.stop_reason,
        )

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            try:
                on_stream = on_stream_end = None
                if msg.metadata.get("_wants_stream"):
                    # Split one answer into distinct stream segments.
                    stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                    stream_segment = 0

                    def _current_stream_id() -> str:
                        return f"{stream_base_id}:{stream_segment}"

                    async def on_stream(delta: str) -> None:
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content=delta,
                            metadata={
                                "_stream_delta": True,
                                "_stream_id": _current_stream_id(),
                            },
                        ))

                    async def on_stream_end(*, resuming: bool = False) -> None:
                        nonlocal stream_segment
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="",
                            metadata={
                                "_stream_end": True,
                                "_resuming": resuming,
                                "_stream_id": _current_stream_id(),
                            },
                        ))
                        stream_segment += 1

                response = await self._process_message(
                    msg, on_stream=on_stream, on_stream_end=on_stream_end,
                )
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def _schedule_post_turn_maintenance(
        self,
        session: Session,
        *,
        start_index: int | None = None,
    ) -> None:
        """后台执行回复后的记忆提取和上下文压缩检查。"""
        self._schedule_background(self._post_turn_maintenance(session, start_index=start_index))

    async def _post_turn_maintenance(
        self,
        session: Session,
        *,
        start_index: int | None = None,
    ) -> None:
        """串行执行后台维护，避免多个任务同时保存同一个 session。"""
        try:
            memory_agent = getattr(self, "memory_agent", None) or self.memory_extractor
            await memory_agent.extract_new(
                session,
                self.sessions,
                start_index=start_index,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Post-turn memory extraction failed")

        try:
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Post-turn token consolidation failed")

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(
                channel,
                chat_id,
                msg.metadata.get("message_id"),
                sender_id=msg.sender_id,
                session_key=key,
            )
            history = session.get_history(max_messages=0)
            current_role = "assistant" if msg.sender_id == "subagent" else "user"
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                current_role=current_role,
                session_summary=session.metadata.get(
                    self.memory_consolidator.SESSION_SUMMARY_KEY
                ),
                harness_state=format_harness_state(session.metadata),
            )
            run_result = await self._run_agent_loop(
                messages, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
                session_key=key,
            )
            final_content, _, all_msgs = run_result
            previous_message_count = len(session.messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self._update_harness_state_after_save(session, previous_message_count)
            self.sessions.save(session)
            self._schedule_post_turn_maintenance(session, start_index=previous_message_count)
            if run_result.stop_reason == "approval_required" and run_result.approval_event:
                return self._approval_outbound(
                    channel=channel,
                    chat_id=chat_id,
                    event=run_result.approval_event,
                    base_metadata=msg.metadata,
                    web_control=channel == "web",
                )
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key

        approval_handled, approval_response = await self._handle_approval_response(msg, key)
        if approval_handled:
            return approval_response

        session = self.sessions.get_or_create(key)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        if shortcut := await self._handle_uploaded_file_index_shortcut(msg, session, raw):
            return shortcut

        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(
            msg.channel,
            msg.chat_id,
            msg.metadata.get("message_id"),
            sender_id=msg.sender_id,
            session_key=key,
        )
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            session_summary=session.metadata.get(
                self.memory_consolidator.SESSION_SUMMARY_KEY
            ),
            harness_state=format_harness_state(session.metadata),
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        run_result = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=msg.channel, chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            session_key=key,
        )
        final_content, _, all_msgs = run_result

        if run_result.stop_reason == "approval_required" and run_result.approval_event:
            previous_message_count = len(session.messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self._update_harness_state_after_save(session, previous_message_count)
            self.sessions.save(session)
            return self._approval_outbound(
                channel=msg.channel,
                chat_id=msg.chat_id,
                event=run_result.approval_event,
                base_metadata=msg.metadata,
                web_control=msg.channel == "web",
            )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        previous_message_count = len(session.messages)
        self._save_turn(session, all_msgs, 1 + len(history))
        self._update_harness_state_after_save(session, previous_message_count)
        self.sessions.save(session)
        self._schedule_post_turn_maintenance(session, start_index=previous_message_count)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None:
            meta["_streamed"] = True
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=meta,
        )

    async def _handle_uploaded_file_index_shortcut(
        self,
        msg: InboundMessage,
        session: Session,
        raw: str,
    ) -> OutboundMessage | None:
        """承接“索引刚上传的文件”和上一轮索引确认短回复。"""
        wants_index = self._is_index_latest_upload_request(raw)
        confirms_index = self._is_short_affirmative(raw) and self._recently_asked_to_index_upload(session)
        if not wants_index and not confirms_index:
            return None

        file_path = self._find_recent_upload_file(session, msg.channel)
        if file_path is None:
            return None

        arguments = {"file_path": str(file_path)}
        approval_event = self.approvals.request(
            session_key=msg.session_key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            tool_name="index_document",
            arguments=arguments,
        )
        if approval_event:
            previous_message_count = len(session.messages)
            session.add_message("user", msg.content)
            outbound = self._approval_outbound(
                channel=msg.channel,
                chat_id=msg.chat_id,
                event=approval_event,
                base_metadata=msg.metadata,
                web_control=msg.channel == "web",
            )
            if outbound.content:
                session.add_message("assistant", outbound.content)
            self._update_harness_state_after_save(session, previous_message_count)
            self.sessions.save(session)
            return outbound

        self._set_tool_context(
            msg.channel,
            msg.chat_id,
            msg.metadata.get("message_id"),
            sender_id=msg.sender_id,
            session_key=msg.session_key,
        )
        logger.info("Uploaded file index shortcut: {}", file_path)
        result = await self.tools.execute("index_document", arguments)
        content = self._format_index_shortcut_result(file_path, result)

        previous_message_count = len(session.messages)
        session.add_message("user", msg.content)
        session.add_message("assistant", content)
        self._update_harness_state_after_save(session, previous_message_count)
        self.sessions.save(session)
        self._schedule_post_turn_maintenance(session, start_index=previous_message_count)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=content,
            metadata=dict(msg.metadata or {}),
        )

    @classmethod
    def _is_short_affirmative(cls, raw: str) -> bool:
        """判断短回复是否是明确确认。"""
        normalized = raw.strip().strip("。！!,.， ")
        return len(normalized) <= 8 and normalized in cls._SHORT_AFFIRMATIVES

    @staticmethod
    def _is_index_latest_upload_request(raw: str) -> bool:
        """判断用户是否明确要求索引刚上传的文件。"""
        text = raw.strip()
        return "索引" in text and any(
            marker in text for marker in ("刚才", "上传", "这个文件", "这份文件", "刚发", "刚传")
        )

    def _recently_asked_to_index_upload(self, session: Session) -> bool:
        """判断最近助手消息是否在询问是否索引上传文件。"""
        for message in reversed(session.messages[-8:]):
            if message.get("role") != "assistant":
                continue
            text = self._assistant_visible_or_message_tool_text(message)
            if "索引" in text and ("RAG" in text or "知识库" in text) and ("确认" in text or "回复" in text):
                return True
        return False

    def _find_recent_upload_file(self, session: Session, channel: str) -> Path | None:
        """从最近会话或渠道 media 目录中找到最近上传的可索引文件。"""
        for message in reversed(session.messages[-30:]):
            content = self._history_text(message.get("content"))
            for candidate in self._extract_uploaded_file_paths(content):
                if self._is_indexable_file(candidate):
                    return candidate
            for media in message.get("media") or []:
                candidate = Path(str(media))
                if self._is_indexable_file(candidate):
                    return candidate

        media_dir = get_default_data_dir() / "media" / channel
        if media_dir.is_dir():
            files = [
                item for item in media_dir.iterdir()
                if item.is_file() and item.suffix.lower() in self._INDEXABLE_UPLOAD_SUFFIXES
            ]
            if files:
                return max(files, key=lambda item: item.stat().st_mtime)
        return None

    @classmethod
    def _is_indexable_file(cls, path: Path) -> bool:
        """判断路径是否是存在的可索引文件。"""
        return path.is_file() and path.suffix.lower() in cls._INDEXABLE_UPLOAD_SUFFIXES

    @staticmethod
    def _extract_uploaded_file_paths(text: str) -> list[Path]:
        """从渠道上传文件标记中提取文件路径。"""
        if not text:
            return []
        matches = re.findall(r"\[File:\s*source:\s*([^\]\n\r]+)\]", text, flags=re.I)
        return [Path(match.strip().strip('"')) for match in matches]

    @staticmethod
    def _assistant_visible_or_message_tool_text(message: dict[str, Any]) -> str:
        """读取普通助手文本或 message 工具发送给用户的文本。"""
        content = AgentLoop._history_text(message.get("content"))
        parts = [content] if content else []
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict) or function.get("name") != "message":
                continue
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            text = args.get("content")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts)

    @staticmethod
    def _format_index_shortcut_result(file_path: Path, raw_result: Any) -> str:
        """把索引工具结果转成用户可读回复。"""
        try:
            payload = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        except (TypeError, json.JSONDecodeError):
            payload = None

        if isinstance(payload, dict):
            if payload.get("success"):
                chunks = payload.get("chunk_count", 0)
                if payload.get("skipped"):
                    return f"{file_path.name} 已经在知识库中，无需重复索引。"
                return f"已把 {file_path.name} 索引到知识库，共 {chunks} 个分块。现在可以直接问我这份文档的内容。"
            error = payload.get("user_message") or payload.get("error")
            if error:
                return f"{file_path.name} 索引失败：{error}"
        return f"{file_path.name} 索引完成。"

    async def _handle_approval_response(
        self,
        msg: InboundMessage,
        session_key: str,
    ) -> tuple[bool, OutboundMessage | None]:
        """处理用户对待审核操作的同意或拒绝。"""
        resolution = self.approvals.consume_response(session_key, msg.content)
        if resolution is None:
            return False, None
        if resolution.request is None:
            return True, OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=resolution.message,
                metadata=dict(msg.metadata or {}),
            )

        if not resolution.request.resume_messages or not resolution.request.tool_calls:
            if resolution.status not in {"approved", "approved_temporary"}:
                return True, OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=resolution.message,
                    metadata=dict(msg.metadata or {}),
                )
            result = await self.tools.execute(
                resolution.request.tool_name,
                resolution.request.arguments,
            )
            return True, OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"已执行确认操作：{resolution.request.summary}\n\n结果：{result}",
                metadata=dict(msg.metadata or {}),
            )

        return True, await self._continue_after_approval(msg, session_key, resolution)

    def _approval_outbound(
        self,
        *,
        channel: str,
        chat_id: str,
        event: dict[str, Any],
        base_metadata: dict[str, Any] | None,
        web_control: bool,
    ) -> OutboundMessage:
        """生成审批等待消息，Web 用控制事件，其他渠道用文字。"""
        meta = dict(base_metadata or {})
        approval_id = str(event.get("approval_id") or "")
        approval_meta = self.approvals.metadata_for_request(approval_id)
        if approval_meta:
            meta["approval"] = approval_meta
        if web_control:
            meta["_control"] = "approval_required"
        content = "" if web_control else str(event.get("content") or "需要你确认。")
        return OutboundMessage(channel=channel, chat_id=chat_id, content=content, metadata=meta)

    async def _continue_after_approval(
        self,
        msg: InboundMessage,
        session_key: str,
        resolution: Any,
    ) -> OutboundMessage | None:
        """把审批结果接回原来的工具调用流程。"""
        if resolution.request is None:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="当前没有匹配的待确认操作。",
                metadata=dict(msg.metadata or {}),
            )

        approval = resolution.request
        self._set_tool_context(
            msg.channel,
            msg.chat_id,
            msg.metadata.get("message_id"),
            sender_id=msg.sender_id,
            session_key=session_key,
        )

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        messages = [dict(message) for message in approval.resume_messages]
        messages.append(build_assistant_message(
            approval.response_content or "",
            tool_calls=[tool_call.to_openai_tool_call() for tool_call in approval.tool_calls],
            reasoning_content=approval.reasoning_content,
            thinking_blocks=approval.thinking_blocks,
        ))
        for tool_call in approval.tool_calls:
            result = await self._approval_tool_result(
                tool_call,
                approved=resolution.status in {"approved", "approved_temporary"},
                approved_request=approval,
                channel=msg.channel,
                chat_id=msg.chat_id,
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": result,
            })

        run_result = await self._run_agent_loop(
            messages,
            channel=msg.channel,
            chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            session_key=session_key,
            tools_enabled=resolution.status == "approved",
        )
        final_content, _, all_msgs = run_result
        session = self.sessions.get_or_create(session_key)
        previous_message_count = len(session.messages)
        self._save_turn(session, all_msgs, approval.resume_save_skip)
        self._update_harness_state_after_save(session, previous_message_count)
        self.sessions.save(session)
        self._schedule_post_turn_maintenance(session, start_index=previous_message_count)

        if run_result.stop_reason == "approval_required" and run_result.approval_event:
            return self._approval_outbound(
                channel=msg.channel,
                chat_id=msg.chat_id,
                event=run_result.approval_event,
                base_metadata=msg.metadata,
                web_control=msg.channel == "web",
            )

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content or "操作已处理。",
            metadata=dict(msg.metadata or {}),
        )

    async def _approval_tool_result(
        self,
        tool_call: ToolCallRequest,
        *,
        approved: bool,
        approved_request: Any,
        channel: str,
        chat_id: str,
    ) -> Any:
        """根据审批结果生成对应的工具结果。"""
        if not approved:
            return f"用户已拒绝执行：{approved_request.summary}。不要执行该动作，请直接回复用户。"

        arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        if tool_call.id != approved_request.tool_call_id:
            decision = self.approvals.policy.evaluate(
                tool_call.name,
                arguments,
                channel=channel,
                chat_id=chat_id,
            )
            if decision is not None:
                return "该工具调用也需要单独人工确认，当前没有执行。"
        try:
            return await self.tools.execute(tool_call.name, arguments)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            return f"Error: {type(exc).__name__}: {exc}"

    @staticmethod
    def _image_placeholder(block: dict[str, Any]) -> dict[str, str]:
        """Convert an inline image block into a compact text placeholder."""
        path = (block.get("_meta") or {}).get("path", "")
        return {"type": "text", "text": f"[image: {path}]" if path else "[image]"}

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if (
                block.get("type") == "image_url"
                and block.get("image_url", {}).get("url", "").startswith("data:image/")
            ):
                filtered.append(self._image_placeholder(block))
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if truncate_text and len(text) > self._TOOL_RESULT_MAX_CHARS:
                    text = text[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def get_session_history(self, session_key: str, limit: int = 100) -> list[dict[str, str]]:
        """返回适合 Web 页面展示的最近会话历史。"""
        try:
            safe_limit = max(0, min(int(100 if limit is None else limit), 100))
        except (TypeError, ValueError):
            safe_limit = 100
        if safe_limit == 0:
            return []
        session = self.sessions.get_or_create(session_key)
        visible: list[dict[str, str]] = []
        for message in session.messages:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            if message.get("tool_calls"):
                continue
            content = self._history_text(message.get("content"))
            if not content:
                continue
            if is_approval_prompt_text(content):
                continue
            visible.append({
                "role": str(role),
                "content": content,
                "timestamp": str(message.get("timestamp") or ""),
            })
        return visible[-safe_limit:]

    @staticmethod
    def _history_text(content: Any) -> str:
        """把 session 中的消息内容转成 Web 历史可展示文本。"""
        if isinstance(content, str):
            text = content
            if text.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                parts = text.split("\n\n", 1)
                text = parts[1] if len(parts) > 1 else ""
            return text.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") in {"text", "input_text", "output_text"}
                    and isinstance(block.get("text"), str)
                ):
                    parts.append(block["text"])
            return "\n".join(part.strip() for part in parts if part.strip()).strip()

        return ""

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "assistant" and is_approval_prompt_text(content):
                continue
            if role == "tool":
                if isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    def _update_harness_state_after_save(self, session: Session, start_index: int) -> None:
        """根据刚保存的消息更新会话恢复状态。"""
        update_session_harness_state(
            session,
            session.messages[start_index:],
            self.workspace,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        return await self._process_message(
            msg, session_key=session_key, on_progress=on_progress,
            on_stream=on_stream, on_stream_end=on_stream_end,
        )
