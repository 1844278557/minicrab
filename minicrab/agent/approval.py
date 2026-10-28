"""人工审核：在高风险工具真正执行前暂停并等待用户确认。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from minicrab.agent.permissions import PermissionDecision, PermissionPolicy, TemporaryPermissionStore


@dataclass(slots=True)
class ApprovalDecision:
    """一次工具调用需要审核时的说明。"""

    summary: str
    reason: str
    action: str = "ask"
    source: str = "default"


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

    def __init__(self, workspace: Path):
        """初始化统一权限策略。"""
        self.permission = PermissionPolicy(workspace)

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        channel: str = "",
        chat_id: str = "",
    ) -> ApprovalDecision | None:
        """判断一次工具调用是否需要人工审核。"""
        decision = self.permission.evaluate(
            tool_name,
            arguments,
            channel=channel,
            chat_id=chat_id,
        )
        if decision.action == "allow":
            return None
        return ApprovalDecision(
            summary=decision.summary,
            reason=decision.reason,
            action=decision.action,
            source=decision.source,
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
        self.temporary_permissions = TemporaryPermissionStore(ttl_seconds=ttl_seconds)
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
        permission = self.policy.permission.evaluate(
            tool_name,
            arguments,
            channel=channel,
            chat_id=chat_id,
        )
        fingerprint = self.policy.permission.fingerprint(tool_name, arguments)
        if permission.action == "deny":
            self.policy.permission.audit(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                tool_name=tool_name,
                arguments=arguments,
                decision=permission,
            )
            return {
                "status": "denied",
                "content": f"权限已拒绝：{permission.reason}",
                "approval_id": "",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "summary": permission.summary,
                "reason": permission.reason,
            }
        if permission.action == "allow":
            self.policy.permission.audit(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                tool_name=tool_name,
                arguments=arguments,
                decision=permission,
            )
            return None
        if self.temporary_permissions.allows(session_key, fingerprint):
            self.policy.permission.audit(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                tool_name=tool_name,
                arguments=arguments,
                decision=PermissionDecision("allow", permission.summary, "命中 30 分钟临时授权。", "temporary"),
                temporary=True,
            )
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
            summary=permission.summary,
            reason=permission.reason,
            fingerprint=fingerprint,
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
        self.policy.permission.audit(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            tool_name=tool_name,
            arguments=arguments,
            decision=permission,
            approval_id=request.id,
        )
        return {
            "status": "approval_required",
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
            self.policy.permission.audit(
                session_key=session_key,
                channel=request.channel,
                chat_id=request.chat_id,
                tool_name=request.tool_name,
                arguments=request.arguments,
                decision=PermissionDecision("deny", request.summary, "用户取消执行。", "user"),
                approval_id=request.id,
            )
            return ApprovalResolution("rejected", f"已取消：{request.summary}", request)

        if action == "approve_temporary":
            self.temporary_permissions.add(session_key, request.fingerprint)
            request.status = "approved_temporary"
            self.policy.permission.audit(
                session_key=session_key,
                channel=request.channel,
                chat_id=request.chat_id,
                tool_name=request.tool_name,
                arguments=request.arguments,
                decision=PermissionDecision("allow", request.summary, "用户允许 30 分钟。", "user"),
                temporary=True,
                approval_id=request.id,
            )
            return ApprovalResolution("approved_temporary", f"已允许 30 分钟：{request.summary}", request)

        request.status = "approved"
        self.policy.permission.audit(
            session_key=session_key,
            channel=request.channel,
            chat_id=request.chat_id,
            tool_name=request.tool_name,
            arguments=request.arguments,
            decision=PermissionDecision("allow", request.summary, "用户确认执行。", "user"),
            approval_id=request.id,
        )
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
        title, details = self._describe_request(request)
        return {
            "id": request.id,
            "toolName": request.tool_name,
            "summary": request.summary,
            "reason": request.reason,
            "title": title,
            "detail": "\n".join(details),
            "allowTemporary": "true",
        }

    def format_prompt(self, request: ApprovalRequest) -> str:
        """生成发给用户的确认文本。"""
        pending = self._pending_for_session(request.session_key)
        if len(pending) > 1:
            return self._format_multiple_prompt(pending)
        return self._format_single_prompt(request)

    def _format_single_prompt(self, request: ApprovalRequest) -> str:
        """生成单个待确认事项的自然语言提示。"""
        title, details = self._describe_request(request)
        lines = [f"要我{title}吗？"]
        if details:
            lines.append("")
            lines.extend(details)
        lines.extend(["", "回复“确认”继续，回复“确认30分钟”临时允许同类操作，回复“取消”放弃。"])
        return "\n".join(lines)

    def _format_multiple_prompt(self, requests: list[ApprovalRequest]) -> str:
        """生成多个待确认事项的自然语言提示。"""
        lines = [f"你有 {len(requests)} 个待确认事项：", ""]
        for index, request in enumerate(requests, start=1):
            title, _ = self._describe_request(request)
            lines.append(f"{index}. {title}")
        lines.extend(["", "回复“确认 1”继续，回复“确认30分钟 1”临时允许同类操作，回复“取消 1”放弃。"])
        return "\n".join(lines)

    def _describe_request(self, request: ApprovalRequest) -> tuple[str, list[str]]:
        """把内部工具调用转换成用户能理解的动作说明。"""
        args = request.arguments
        if request.tool_name == "remember_user_info":
            target = str(args.get("target") or "user").lower()
            if target == "memory":
                return "更新长期工作记忆", ["目标：memory/MEMORY.md", "说明：会影响以后对话中的长期工作记忆。"]
            return "更新长期用户记忆", ["目标：USER.md", "说明：会影响以后对话中的长期用户记忆。"]
        if request.tool_name == "manage_memory":
            target = str(args.get("target") or "user").lower()
            op = str(args.get("op") or "")
            if target == "memory":
                return "更新长期工作记忆", [f"操作：{op}", "目标：memory/MEMORY.md", "说明：会影响以后对话中的长期工作记忆。"]
            return "更新长期用户记忆", [f"操作：{op}", "目标：USER.md", "说明：会影响以后对话中的长期用户记忆。"]
        if request.tool_name == "clear_rag_store":
            return "清空知识库", ["说明：会删除所有已索引文档和向量数据。"]
        if request.tool_name == "remove_rag_document":
            file_path = str(args.get("file_path") or "")
            details = ["说明：会从知识库中移除指定文档。"]
            if file_path:
                details.insert(0, f"文档：{self._short(file_path, 80)}")
            return "删除知识库文档", details
        if request.tool_name == "cleanup_expired_rag":
            return "清理过期知识库文档", ["说明：会删除过期文档的索引数据。"]
        if request.tool_name == "index_document":
            file_path = str(args.get("file_path") or "")
            details = ["说明：会把文档加入知识库，方便后续问答。"]
            if file_path:
                details.insert(0, f"文档：{self._short(file_path, 80)}")
            return "索引这个文档", details
        if request.tool_name == "cron":
            return self._describe_cron_request(args)
        if request.tool_name == "exec":
            command = str(args.get("command") or "")
            details = [f"命令：{self._short(command, 120)}"] if command else []
            if request.reason:
                details.append(f"说明：{request.reason}")
            return "执行这条命令", details
        if request.tool_name in {"write_file", "edit_file"}:
            path = str(args.get("path") or "")
            details = [f"文件：{self._short(path, 100)}"] if path else []
            if request.reason:
                details.append(f"说明：{request.reason}")
            return "修改这个文件", details
        if request.tool_name == "message":
            content = str(args.get("content") or "")
            details = [f"内容：{self._short(content, 120)}"] if content else []
            if request.reason:
                details.append(f"说明：{request.reason}")
            return "发送这条消息", details
        details = [f"说明：{request.reason}"] if request.reason else []
        return request.summary, details

    def _describe_cron_request(self, args: dict[str, Any]) -> tuple[str, list[str]]:
        """生成定时任务的用户说明。"""
        action = str(args.get("action") or "").lower()
        if action == "remove":
            job_id = str(args.get("job_id") or "")
            details = [f"任务：{job_id}"] if job_id else []
            details.append("说明：会删除已有定时任务。")
            return "删除定时任务", details

        message = str(args.get("message") or "")
        details = []
        if message:
            details.append(f"内容：{self._short(message, 120)}")
        if args.get("every_seconds"):
            details.append(f"频率：每 {args['every_seconds']} 秒")
        elif args.get("cron_expr"):
            timing = str(args.get("cron_expr") or "")
            tz = str(args.get("tz") or "")
            details.append(f"时间规则：{timing}{f'（{tz}）' if tz else ''}")
        details.append("说明：会在未来重复或长期执行。")
        return "设置定时任务", details

    def _find_request(
        self,
        session_key: str,
        approval_id: str | None,
    ) -> ApprovalRequest | str | None:
        """找到当前会话中对应的待确认请求。"""
        if approval_id:
            pending = self._pending_for_session(session_key)
            if approval_id.isdigit():
                index = int(approval_id) - 1
                if 0 <= index < len(pending):
                    return pending[index]
                return None
            request = self._requests.get(approval_id)
            if request and request.session_key == session_key:
                return request
            return None

        pending = self._pending_for_session(session_key)
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
        if len(parts) == 1:
            match = re.match(r"^(同意30分钟|确认30分钟|同意|确认|拒绝|取消)(\d+)$", parts[0])
            if match:
                parts = [match.group(1), match.group(2)]
        word = parts[0].lower()
        approval_id = parts[1] if len(parts) > 1 else None
        if word in {"同意30分钟", "确认30分钟", "approve30", "approve-30m"}:
            return "approve_temporary", approval_id
        if word in self._APPROVE_WORDS:
            return "approve", approval_id
        if word in self._REJECT_WORDS:
            return "reject", approval_id
        return None

    def _pending_for_session(self, session_key: str) -> list[ApprovalRequest]:
        """返回当前会话未过期的待确认事项。"""
        pending: list[ApprovalRequest] = []
        for request_id, request in list(self._requests.items()):
            if request.expired():
                self._requests.pop(request_id, None)
                continue
            if request.session_key == session_key and request.status == "pending":
                pending.append(request)
        return sorted(pending, key=lambda item: item.created_at)

    def _next_id(self, now: datetime) -> str:
        """生成用户可读的审批编号。"""
        self._counter += 1
        return f"A-{now:%Y%m%d%H%M%S}-{self._counter:03d}"

    @staticmethod
    def _short(text: str, limit: int = 120) -> str:
        """缩短展示给用户看的长文本。"""
        clean = " ".join(text.split())
        return clean if len(clean) <= limit else clean[:limit] + "..."
