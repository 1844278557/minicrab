"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from minicrab.agent.hook import AgentHook, AgentHookContext
from minicrab.agent.tool_summary import ToolResultSummarizer
from minicrab.agent.tools.registry import ToolRegistry
from minicrab.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from minicrab.utils.helpers import build_assistant_message, contains_tool_call_markup

_DEFAULT_MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)
_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    approval_callback: Callable[..., Any] | None = None
    tools_enabled: bool = True
    stop_after_tools_callback: Callable[[], bool] | None = None


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    approval_event: dict[str, Any] | None = None


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages = list(spec.initial_messages)
        final_content: str | None = None
        tools_used: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []
        approval_event: dict[str, Any] | None = None

        for iteration in range(spec.max_iterations):
            context = AgentHookContext(iteration=iteration, messages=messages)
            await hook.before_iteration(context)
            kwargs: dict[str, Any] = {
                "messages": messages,
                "tools": spec.tools.get_definitions() if spec.tools_enabled else [],
                "model": spec.model,
            }
            if spec.temperature is not None:
                kwargs["temperature"] = spec.temperature
            if spec.max_tokens is not None:
                kwargs["max_tokens"] = spec.max_tokens
            if spec.reasoning_effort is not None:
                kwargs["reasoning_effort"] = spec.reasoning_effort

            if hook.wants_streaming():
                async def _stream(delta: str) -> None:
                    await hook.on_stream(context, delta)

                response = await self.provider.chat_stream_with_retry(
                    **kwargs,
                    on_content_delta=_stream,
                )
            else:
                response = await self.provider.chat_with_retry(**kwargs)

            raw_usage = response.usage or {}
            usage = {
                "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
            }
            context.response = response
            context.usage = usage
            context.tool_calls = list(response.tool_calls)

            if response.has_tool_calls:
                if not spec.tools_enabled:
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=False)
                    clean = hook.finalize_content(context, response.content)
                    if contains_tool_call_markup(clean):
                        clean = None
                    final_content = clean or "操作已取消，未执行。"
                    messages.append(build_assistant_message(
                        final_content,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    ))
                    context.final_content = final_content
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    break

                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)
                tools_used.extend(tc.name for tc in response.tool_calls)

                await hook.before_execute_tools(context)

                approval_event = await self._request_approval(
                    spec,
                    response.tool_calls,
                    messages,
                    response,
                )
                if approval_event:
                    if approval_event.get("status") == "denied":
                        denied_results = self._build_denied_tool_results(
                            response.tool_calls,
                            approval_event,
                        )
                        event = {
                            "name": str(approval_event.get("tool_name") or response.tool_calls[0].name),
                            "status": "permission_denied",
                            "detail": str(approval_event.get("reason") or "权限拒绝"),
                        }
                        tool_events.append(event)
                        context.tool_results = list(denied_results)
                        context.tool_events = [event]
                        messages.append(build_assistant_message(
                            response.content or "",
                            tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                            reasoning_content=response.reasoning_content,
                            thinking_blocks=response.thinking_blocks,
                        ))
                        for tool_call, result in zip(response.tool_calls, denied_results):
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": result,
                            })
                        await hook.after_iteration(context)
                        continue

                    event = {
                        "name": str(approval_event.get("tool_name") or response.tool_calls[0].name),
                        "status": "approval_required",
                        "detail": str(approval_event.get("summary") or "需要人工确认"),
                        "approval_id": str(approval_event.get("approval_id") or ""),
                    }
                    tool_events.append(event)
                    context.tool_results = []
                    context.tool_events = [event]
                    context.final_content = None
                    context.stop_reason = "approval_required"
                    stop_reason = "approval_required"
                    await hook.after_iteration(context)
                    break

                results, new_events, fatal_error = await self._execute_tools(
                    spec,
                    response.tool_calls,
                )
                tool_events.extend(new_events)
                context.tool_events = list(new_events)
                if fatal_error is not None:
                    error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                    stop_reason = "tool_error"
                    context.error = error
                    context.stop_reason = stop_reason
                    context.tool_results = list(results)
                    await hook.after_iteration(context)
                    break
                results = await self._summarize_tool_results(
                    spec,
                    response.tool_calls,
                    results,
                )
                context.tool_results = list(results)
                messages.append(build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                ))
                for tool_call, result in zip(response.tool_calls, results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result,
                    })
                if (
                    spec.stop_after_tools_callback is not None
                    and spec.stop_after_tools_callback()
                ):
                    stop_reason = "tool_completed"
                    context.final_content = None
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    break
                await hook.after_iteration(context)
                continue

            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=False)

            clean = hook.finalize_content(context, response.content)
            if not spec.tools_enabled and contains_tool_call_markup(clean):
                clean = "操作已取消，未执行。"
            if response.finish_reason == "error":
                final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break

            messages.append(build_assistant_message(
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            ))
            final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            break
        else:
            stop_reason = "max_iterations"
            template = spec.max_iterations_message or _DEFAULT_MAX_ITERATIONS_MESSAGE
            final_content = template.format(max_iterations=spec.max_iterations)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
            approval_event=approval_event,
        )

    async def _request_approval(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
        messages: list[dict[str, Any]],
        response: LLMResponse,
    ) -> dict[str, Any] | None:
        """在工具执行前检查是否需要人工确认。"""
        if spec.approval_callback is None:
            return None
        for tool_call in tool_calls:
            event = self._call_approval_callback(
                spec.approval_callback,
                tool_call,
                tool_calls,
                messages,
                response,
            )
            if inspect.isawaitable(event):
                event = await event
            if event:
                return event
        return None

    @staticmethod
    def _build_denied_tool_results(
        tool_calls: list[ToolCallRequest],
        event: dict[str, Any],
    ) -> list[str]:
        """为被权限拒绝的工具调用构造工具结果。"""
        denied_id = str(event.get("tool_call_id") or "")
        reason = str(event.get("reason") or "权限拒绝")
        summary = str(event.get("summary") or "该操作")
        results: list[str] = []
        for tool_call in tool_calls:
            if not denied_id or tool_call.id == denied_id:
                results.append(f"Error: Permission denied: {summary}。原因：{reason}")
            else:
                results.append("Error: Permission denied: 同一轮存在被拒绝的工具调用，本轮其他工具未执行。")
        return results

    @staticmethod
    def _call_approval_callback(
        callback: Callable[..., Any],
        tool_call: ToolCallRequest,
        tool_calls: list[ToolCallRequest],
        messages: list[dict[str, Any]],
        response: LLMResponse,
    ) -> Any:
        """兼容旧的一参回调和新的四参回调。"""
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return callback(tool_call)

        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
        ]
        has_varargs = any(
            parameter.kind == parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        if has_varargs or len(positional) >= 4:
            return callback(tool_call, tool_calls, messages, response)
        return callback(tool_call)

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> tuple[list[Any], list[dict[str, str]], BaseException | None]:
        if not spec.concurrent_tools:
            tool_results = [
                await self._run_tool(spec, tool_call)
                for tool_call in tool_calls
            ]
        else:
            tool_results = await self._execute_tools_safely_parallel(spec, tool_calls)

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        for result, event, error in tool_results:
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
        return results, events, fatal_error

    async def _execute_tools_safely_parallel(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> list[tuple[Any, dict[str, str], BaseException | None]]:
        """只并行连续的只读/查询工具；有副作用工具保持顺序执行。"""
        tool_results: list[tuple[Any, dict[str, str], BaseException | None]] = []
        parallel_group: list[ToolCallRequest] = []

        async def flush_parallel_group() -> None:
            if not parallel_group:
                return
            group = list(parallel_group)
            parallel_group.clear()
            tool_results.extend(await asyncio.gather(*(
                self._run_tool(spec, tool_call)
                for tool_call in group
            )))

        for tool_call in tool_calls:
            if self._is_parallel_safe_tool(tool_call):
                parallel_group.append(tool_call)
                continue
            await flush_parallel_group()
            tool_results.append(await self._run_tool(spec, tool_call))
        await flush_parallel_group()
        return tool_results

    @staticmethod
    def _is_parallel_safe_tool(tool_call: ToolCallRequest) -> bool:
        """判断工具是否适合和同一轮其他只读工具并行。"""
        read_only_tools = {
            "read_file",
            "list_dir",
            "web_search",
            "web_fetch",
            "search_rag",
            "list_rag_documents",
            "search_episodes",
            "list_recent_episodes",
            "get_episode_stats",
        }
        if tool_call.name in read_only_tools:
            return True
        if tool_call.name == "cron":
            action = str((tool_call.arguments or {}).get("action") or "list").lower()
            return action in {"", "list"}
        return False

    async def _summarize_tool_results(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
        results: list[Any],
    ) -> list[Any]:
        """摘要超长工具结果；这是内部压缩调用，不是 Agent 行为。"""
        summarizer = ToolResultSummarizer(self.provider, spec.model)
        summarized: list[Any] = []
        for tool_call, result in zip(tool_calls, results):
            summarized.append(await summarizer.summarize_if_needed(tool_call, result))
        return summarized

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
    ) -> tuple[Any, dict[str, str], BaseException | None]:
        try:
            result = await spec.tools.execute(tool_call.name, tool_call.arguments)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc
            return f"Error: {type(exc).__name__}: {exc}", event, None

        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        return result, {
            "name": tool_call.name,
            "status": "error" if isinstance(result, str) and result.startswith("Error") else "ok",
            "detail": detail,
        }, None
