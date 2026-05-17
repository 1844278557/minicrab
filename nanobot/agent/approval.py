"""人工审核：在高风险工具真正执行前暂停并等待用户确认。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ApprovalDecision:
    """一次工具调用需要审核时的说明。"""

    summary: str
    reason: str


@dataclass(slots=True)
class ApprovalRequest:
    """等待用户确认的一次工具调用。"""

    id: str
    session_key: str
    channel: str
    chat_id: str
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    summary: str
    reason: str
    fingerprint: str
    status: str
    created_at: datetime
    expires_at: datetime
    resume_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[Any] = field(default_factory=list)
    response_content: str | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[dict[str, Any]] | None = None
    resume_save_skip: int = 0

    def expired(self) -> bool:
        """判断审批是否已经过期。"""
        return datetime.now() >= self.expires_at


@dataclass(slots=True)
class ApprovalResolution:
    """用户审批回复的处理结果。"""

    status: str
    message: str
    request: ApprovalRequest | None = None


class ApprovalPolicy:
    """判断哪些工具调用必须人工审核。"""

    _APPROVAL_TOOLS = {
        "remember_user_info": ApprovalDecision(
            summary="写入长期用户记忆",
            reason="会把信息长期保存到 USER.md。",
        ),
        "clear_rag_store": ApprovalDecision(
            summary="清空 RAG 知识库",
            reason="会删除所有已索引文档和向量数据。",
        ),
        "remove_rag_document": ApprovalDecision(
            summary="删除 RAG 文档",
            reason="会从知识库中移除指定文档数据。",
        ),
        "cleanup_expired_rag": ApprovalDecision(
            summary="清理过期 RAG 文档",
            reason="会删除过期文档的索引数据。",
        ),
    }

    _HIGH_RISK_COMMANDS = [
        (re.compile(r"\bgit\s+push\b", re.I), "会把本地提交推送到远端仓库。"),
        (re.compile(r"\bgh\s+release\b", re.I), "会创建或发布 GitHub Release。"),
        (re.compile(r"\bgit\s+reset\s+--hard\b", re.I), "会强制回滚本地代码。"),
        (re.compile(r"\bgit\s+clean\b", re.I), "会删除未跟踪文件。"),
        (re.compile(r"\b(remove-item|rm|del|rmdir)\b", re.I), "会删除文件或目录。"),
        (re.compile(r"\b(pip|uv|npm|pnpm|yarn|winget|brew|apt|apt-get)\s+(install|add|upgrade|update)\b", re.I), "会安装或更新本地环境。"),
        (re.compile(r"\bpython(?:\.exe)?\s+-m\s+pip\s+install\b", re.I), "会安装 Python 依赖。"),
        (re.compile(r"\b(set-executionpolicy|stop-process|taskkill|kill)\b", re.I), "会修改系统执行策略或结束进程。"),
        (re.compile(r"\b(stop-service|restart-service|sc\.exe|net\s+stop)\b", re.I), "会控制系统服务。"),
        (re.compile(r"\b(format|diskpart|mkfs)\b", re.I), "会修改磁盘或文件系统。"),
    ]

    _SENSITIVE_NAME_PARTS = ("token", "secret", "credential", "apikey", "api_key")

    def __init__(self, workspace: Path):
        """保存工作区路径，用于判断文件是否越界。"""
        try:
            self.workspace = Path(workspace).expanduser().resolve()
        except TypeError:
            self.workspace = Path.cwd().resolve()

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        channel: str = "",
        chat_id: str = "",
    ) -> ApprovalDecision | None:
        """判断一次工具调用是否需要人工审核。"""
        if tool_name in self._APPROVAL_TOOLS:
            return self._APPROVAL_TOOLS[tool_name]
        if tool_name == "exec":
            return self._evaluate_exec(arguments)
        if tool_name in {"write_file", "edit_file"}:
            return self._evaluate_write_path(arguments)
        if tool_name == "cron":
            return self._evaluate_cron(arguments)
        if tool_name == "message":
            return self._evaluate_message(arguments, channel=channel, chat_id=chat_id)
        return None

    def _evaluate_exec(self, arguments: dict[str, Any]) -> ApprovalDecision | None:
        """判断命令是否高风险。"""
        command = str(arguments.get("command") or "")
        for pattern, reason in self._HIGH_RISK_COMMANDS:
            if pattern.search(command):
                return ApprovalDecision(summary=f"执行命令：{self._short(command)}", reason=reason)
        return None

    def _evaluate_write_path(self, arguments: dict[str, Any]) -> ApprovalDecision | None:
        """判断文件写入是否越界或触碰敏感文件。"""
        raw_path = arguments.get("path")
        if not raw_path:
            return None

        try:
            path = Path(str(raw_path)).expanduser()
            if not path.is_absolute():
                path = self.workspace / path
            resolved = path.resolve(strict=False)
        except Exception:
            return ApprovalDecision(summary=f"修改文件：{raw_path}", reason="文件路径无法可靠解析。")

        if not self._is_under_workspace(resolved):
            return ApprovalDecision(
                summary=f"修改工作区外文件：{resolved}",
                reason="目标路径不在当前工作区内。",
            )
        if self._is_sensitive_path(resolved):
            return ApprovalDecision(
                summary=f"修改敏感文件：{resolved.name}",
                reason="目标文件可能包含配置、密钥或账号凭证。",
            )
        return None

    def _evaluate_cron(self, arguments: dict[str, Any]) -> ApprovalDecision | None:
        """判断定时任务变更是否需要确认。"""
        action = str(arguments.get("action") or "").lower()
        if action in {"add", "remove"}:
            return ApprovalDecision(
                summary=f"变更定时任务：{action}",
                reason="定时任务会在未来自动执行或被删除。",
            )
        return None

    def _evaluate_message(
        self,
        arguments: dict[str, Any],
        *,
        channel: str,
        chat_id: str,
    ) -> ApprovalDecision | None:
        """判断消息发送是否是高风险外发。"""
        media = arguments.get("media") or []
        target_channel = str(arguments.get("channel") or channel)
        target_chat_id = str(arguments.get("chat_id") or chat_id)
        if media:
            return ApprovalDecision(summary="发送带附件的消息", reason="会把本地文件发送到聊天渠道。")
        if target_channel != channel or target_chat_id != chat_id:
            return ApprovalDecision(
                summary=f"跨会话发送消息到 {target_channel}:{target_chat_id}",
                reason="会向当前聊天之外的位置主动发送内容。",
            )
        return None

    def _is_under_workspace(self, path: Path) -> bool:
        """判断路径是否在工作区内。"""
        return path == self.workspace or self.workspace in path.parents

    def _is_sensitive_path(self, path: Path) -> bool:
        """判断路径是否看起来像敏感配置或密钥文件。"""
        lower_parts = [part.lower() for part in path.parts]
        name = path.name.lower()
        joined = "/".join(lower_parts)
        return (
            ".ssh" in lower_parts
            or name == ".env"
            or name.startswith(".env.")
            or name == "config.json"
            or any(part in joined for part in self._SENSITIVE_NAME_PARTS)
        )

    @staticmethod
    def _short(text: str, limit: int = 120) -> str:
        """缩短展示给用户看的长文本。"""
        clean = " ".join(text.split())
        return clean if len(clean) <= limit else clean[:limit] + "..."


class ApprovalManager:
    """管理待确认请求和用户同意/拒绝回复。"""

    _APPROVE_WORDS = {"同意", "确认", "yes", "y", "ok", "approve"}
    _REJECT_WORDS = {"拒绝", "取消", "no", "n", "reject"}

    def __init__(self, workspace: Path, ttl_seconds: int = 1800):
        """初始化审批管理器。"""
        self.policy = ApprovalPolicy(workspace)
        self.ttl = timedelta(seconds=ttl_seconds)
        self._requests: dict[str, ApprovalRequest] = {}
        self._counter = 0

    def request(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str = "",
        resume_messages: list[dict[str, Any]] | None = None,
        tool_calls: list[Any] | None = None,
        response_content: str | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict[str, Any]] | None = None,
        resume_save_skip: int = 0,
    ) -> dict[str, Any] | None:
        """如果工具调用需要审核，创建待确认请求。"""
        decision = self.policy.evaluate(
            tool_name,
            arguments,
            channel=channel,
            chat_id=chat_id,
        )
        if decision is None:
            return None

        now = datetime.now()
        request = ApprovalRequest(
            id=self._next_id(now),
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            summary=decision.summary,
            reason=decision.reason,
            fingerprint=self._fingerprint(tool_name, arguments),
            status="pending",
            created_at=now,
            expires_at=now + self.ttl,
            resume_messages=[dict(message) for message in (resume_messages or [])],
            tool_calls=list(tool_calls or []),
            response_content=response_content,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
            resume_save_skip=resume_save_skip,
        )
        self._requests[request.id] = request
        return {
            "content": self.format_prompt(request),
            "approval_id": request.id,
            "tool_name": tool_name,
            "summary": request.summary,
            "reason": request.reason,
        }

    def consume_response(self, session_key: str, text: str) -> ApprovalResolution | None:
        """处理用户发来的同意/拒绝文本。"""
        parsed = self._parse_response(text)
        if parsed is None:
            return None

        action, approval_id = parsed
        request = self._find_request(session_key, approval_id)
        if request is None:
            return ApprovalResolution("missing", "当前没有匹配的待确认操作。")
        if request == "ambiguous":
            return ApprovalResolution("ambiguous", "当前有多个待确认操作，请带上确认编号。")
        if request.expired():
            self._requests.pop(request.id, None)
            return ApprovalResolution("expired", f"确认请求 {request.id} 已过期，操作未执行。")

        self._requests.pop(request.id, None)
        if action == "reject":
            request.status = "rejected"
            return ApprovalResolution("rejected", f"已取消：{request.summary}", request)

        request.status = "approved"
        return ApprovalResolution("approved", f"已确认：{request.summary}", request)

    def metadata_from_content(self, session_key: str, content: str | None) -> dict[str, str] | None:
        """从确认消息正文中提取 Web 按钮需要的元数据。"""
        if not content:
            return None
        match = re.search(r"编号：\s*(A-[A-Za-z0-9-]+)", content)
        if not match:
            return None
        request = self._requests.get(match.group(1))
        if not request or request.session_key != session_key or request.expired():
            return None
        return {
            "id": request.id,
            "toolName": request.tool_name,
            "summary": request.summary,
            "reason": request.reason,
        }

    def metadata_for_request(self, approval_id: str) -> dict[str, str] | None:
        """返回 Web 弹窗需要的审批元数据。"""
        request = self._requests.get(approval_id)
        if not request or request.expired():
            return None
        return {
            "id": request.id,
            "toolName": request.tool_name,
            "summary": request.summary,
            "reason": request.reason,
        }

    def format_prompt(self, request: ApprovalRequest) -> str:
        """生成发给用户的确认文本。"""
        return (
            "需要你确认：\n\n"
            f"编号：{request.id}\n"
            f"操作：{request.summary}\n"
            f"原因：{request.reason}\n\n"
            "回复：\n"
            f"同意 {request.id}\n"
            f"拒绝 {request.id}"
        )

    def _find_request(
        self,
        session_key: str,
        approval_id: str | None,
    ) -> ApprovalRequest | str | None:
        """找到当前会话中对应的待确认请求。"""
        if approval_id:
            request = self._requests.get(approval_id)
            if request and request.session_key == session_key:
                return request
            return None

        pending = [
            request
            for request in self._requests.values()
            if request.session_key == session_key and request.status == "pending"
        ]
        if len(pending) == 1:
            return pending[0]
        if len(pending) > 1:
            return "ambiguous"
        return None

    def _parse_response(self, text: str) -> tuple[str, str | None] | None:
        """解析同意/拒绝回复。"""
        parts = text.strip().split()
        if not parts:
            return None
        word = parts[0].lower()
        approval_id = parts[1] if len(parts) > 1 else None
        if word in self._APPROVE_WORDS:
            return "approve", approval_id
        if word in self._REJECT_WORDS:
            return "reject", approval_id
        return None

    def _next_id(self, now: datetime) -> str:
        """生成用户可读的审批编号。"""
        self._counter += 1
        return f"A-{now:%Y%m%d%H%M%S}-{self._counter:03d}"

    @staticmethod
    def _fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
        """生成工具调用指纹，避免确认内容和执行内容混淆。"""
        payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        return f"{tool_name}:{payload}"
