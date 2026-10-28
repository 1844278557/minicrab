"""内部工具结果压缩器：只摘要长工具输出，不执行任何 Agent 行为。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from loguru import logger

from minicrab.providers.base import LLMProvider, ToolCallRequest
from minicrab.utils.helpers import contains_tool_call_markup, strip_think


@dataclass(slots=True)
class ToolResultSummarizer:
    """用一次无工具 LLM 调用压缩超长工具结果。"""

    provider: LLMProvider
    model: str
    threshold_chars: int = 12_000
    max_input_chars: int = 48_000
    head_chars: int = 3_000
    tail_chars: int = 3_000
    max_summary_tokens: int = 700

    async def summarize_if_needed(
        self,
        tool_call: ToolCallRequest,
        result: Any,
    ) -> Any:
        """短结果原样返回；长文本结果摘要失败时退回头尾截断。"""
        if not isinstance(result, str) or len(result) <= self.threshold_chars:
            return result

        fallback = self._fallback_result(tool_call, result)
        try:
            summary = await self._summarize(tool_call, result)
        except Exception:
            logger.exception("Tool result summarization failed")
            return fallback

        if not summary:
            return fallback

        return self._format_summary_result(tool_call, result, summary)

    async def _summarize(self, tool_call: ToolCallRequest, result: str) -> str:
        """调用模型生成摘要；不传工具，不允许继续执行动作。"""
        response = await self.provider.chat_with_retry(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是内部工具结果摘要器，只能压缩工具输出文本。"
                        "不要调用工具，不要执行任务，不要给用户写回复，不要编造。"
                        "保留对主 Agent 下一步判断有用的信息：成功/失败状态、关键错误、"
                        "文件路径、命令结果、测试统计、数量统计和必要上下文。"
                        "输出简洁中文摘要，最多 12 条要点。"
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_summary_prompt(tool_call, result),
                },
            ],
            tools=[],
            model=self.model,
            max_tokens=self.max_summary_tokens,
            temperature=0,
            reasoning_effort=None,
            tool_choice="none",
        )

        if response.finish_reason == "error" or response.has_tool_calls:
            return ""
        summary = strip_think(response.content or "").strip()
        if contains_tool_call_markup(summary):
            return ""
        return summary

    def _build_summary_prompt(self, tool_call: ToolCallRequest, result: str) -> str:
        """构造摘要输入，超大结果只给头尾片段。"""
        arguments = json.dumps(tool_call.arguments, ensure_ascii=False)
        if len(arguments) > 2_000:
            arguments = arguments[:2_000] + "\n... (arguments truncated)"

        source = result
        omitted = 0
        if len(source) > self.max_input_chars:
            keep_each = self.max_input_chars // 2
            omitted = len(source) - self.max_input_chars
            source = (
                source[:keep_each]
                + f"\n\n... ({omitted:,} chars omitted before summarization) ...\n\n"
                + source[-keep_each:]
            )

        return (
            f"Tool name: {tool_call.name}\n"
            f"Tool arguments:\n{arguments}\n\n"
            f"Original result length: {len(result):,} chars\n"
            f"Input omitted before summary: {omitted:,} chars\n\n"
            "Tool result:\n"
            f"{source}"
        )

    def _format_summary_result(
        self,
        tool_call: ToolCallRequest,
        result: str,
        summary: str,
    ) -> str:
        """返回摘要加原文头尾，避免摘要漏掉关键细节。"""
        return (
            "[LLM Tool Result Summary]\n"
            f"Tool: {tool_call.name}\n"
            f"Original length: {len(result):,} chars\n\n"
            f"{summary.strip()}\n\n"
            + self._original_excerpt(result)
        )

    def _fallback_result(self, tool_call: ToolCallRequest, result: str) -> str:
        """模型摘要不可用时使用确定性头尾截断。"""
        return (
            "[Tool Result Truncated]\n"
            f"Tool: {tool_call.name}\n"
            f"Original length: {len(result):,} chars\n"
            "LLM summary unavailable; keeping original head and tail.\n\n"
            + self._original_excerpt(result)
        )

    def _original_excerpt(self, result: str) -> str:
        """保留原始输出头尾片段。"""
        if len(result) <= self.head_chars + self.tail_chars:
            return "[Original]\n" + result
        omitted = len(result) - self.head_chars - self.tail_chars
        return (
            "[Original Head]\n"
            f"{result[:self.head_chars]}\n\n"
            f"... ({omitted:,} chars omitted) ...\n\n"
            "[Original Tail]\n"
            f"{result[-self.tail_chars:]}"
        )
