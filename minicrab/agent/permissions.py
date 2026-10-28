"""统一工具权限判断、临时授权和审计记录。"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from minicrab.security.network import contains_internal_url
from minicrab.utils.helpers import ensure_dir


@dataclass(slots=True)
class PermissionDecision:
    """一次工具调用的权限判断结果。"""

    action: str
    summary: str
    reason: str
    source: str = "default"


@dataclass(slots=True)
class TemporaryGrant:
    """短时间内允许同类工具调用。"""

    session_key: str
    fingerprint: str
    expires_at: datetime


class PermissionPolicy:
    """统一判断工具调用是允许、询问还是禁止。"""

    _READ_TOOLS = {
        "list_dir",
        "web_search",
        "search_rag",
        "list_rag_documents",
        "search_episodes",
        "list_recent_episodes",
        "get_episode_stats",
    }
    _RAG_DELETE_TOOLS = {
        "clear_rag_store",
        "remove_rag_document",
        "cleanup_expired_rag",
    }
    _SENSITIVE_NAME_PARTS = ("token", "secret", "credential", "apikey", "api_key")
    _WRITE_NAME_HINTS = (
        "write",
        "edit",
        "update",
        "delete",
        "remove",
        "clear",
        "send",
        "create",
        "post",
        "publish",
        "upload",
    )

    _DENY_COMMANDS = (
        (re.compile(r"\b(format|diskpart|mkfs)\b", re.I), "会修改磁盘或文件系统。"),
        (re.compile(r"\bdd\s+if=", re.I), "可能直接写入磁盘设备。"),
        (re.compile(r">\s*/dev/sd", re.I), "可能直接写入磁盘设备。"),
        (re.compile(r"\b(shutdown|reboot|poweroff)\b", re.I), "会关闭或重启系统。"),
        (re.compile(r":\(\)\s*\{.*\};\s*:", re.I), "疑似 fork bomb。"),
        (re.compile(r"\bRemove-Item\s+['\"]?(?:[A-Za-z]:\\|/)\s+-Recurse\b", re.I), "会递归删除根级目录。"),
        (re.compile(r"\brm\s+-rf\s+['\"]?/(?:\s|$)", re.I), "会删除系统根目录。"),
    )
    _ASK_COMMANDS = (
        (re.compile(r"\bgit\s+push\b", re.I), "会把本地提交推送到远端仓库。"),
        (re.compile(r"\bgh\s+release\b", re.I), "会创建或发布 GitHub Release。"),
        (re.compile(r"\bgit\s+reset\s+--hard\b", re.I), "会强制回滚本地代码。"),
        (re.compile(r"\bgit\s+clean\b", re.I), "会删除未跟踪文件。"),
        (re.compile(r"\b(remove-item|rm|del|rmdir)\b", re.I), "会删除文件或目录。"),
        (re.compile(r"\b(pip|uv|npm|pnpm|yarn|winget|brew|apt|apt-get)\s+(install|add|upgrade|update)\b", re.I), "会安装或更新本地环境。"),
        (re.compile(r"\bpython(?:\.exe)?\s+-m\s+pip\s+install\b", re.I), "会安装 Python 依赖。"),
        (re.compile(r"\b(set-executionpolicy|stop-process|taskkill|kill)\b", re.I), "会修改系统执行策略或结束进程。"),
        (re.compile(r"\b(stop-service|restart-service|sc\.exe|net\s+stop)\b", re.I), "会控制系统服务。"),
        (re.compile(r"\b(start-process)\b", re.I), "会启动本地进程。"),
        (re.compile(r"\b(invoke-webrequest|invoke-restmethod|iwr|irm|curl|wget)\b", re.I), "会从网络请求或下载内容。"),
    )

    def __init__(self, workspace: Path):
        """保存工作区路径并读取用户自定义权限规则。"""
        self.workspace = Path(workspace).expanduser().resolve()
        self.config_file = self.workspace / "permissions.json"
        self.audit_file = self.workspace / "memory" / "permissions.log.jsonl"
        self.rules = self._load_rules()

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        channel: str = "",
        chat_id: str = "",
    ) -> PermissionDecision:
        """返回工具调用权限判断。"""
        configured = self._evaluate_config_rules(tool_name, arguments)
        if configured is not None:
            return configured

        if tool_name == "exec":
            return self._evaluate_exec(arguments)
        if tool_name == "read_file":
            return self._evaluate_read_path(arguments)
        if tool_name in {"write_file", "edit_file"}:
            return self._evaluate_write_path(tool_name, arguments)
        if tool_name in {"remember_user_info", "manage_memory"}:
            return self._evaluate_memory_update(arguments)
        if tool_name == "cron":
            return self._evaluate_cron(arguments)
        if tool_name == "message":
            return self._evaluate_message(arguments, channel=channel, chat_id=chat_id)
        if tool_name in self._RAG_DELETE_TOOLS:
            return self._ask(self._rag_summary(tool_name), self._rag_reason(tool_name))
        if tool_name == "spawn":
            return self._ask("启动子任务", "会让子 agent 继续执行一段独立任务。")
        if tool_name == "web_fetch":
            return self._evaluate_web_fetch(arguments)
        if tool_name == "index_document":
            return self._allow("索引文档", "文档索引是可恢复的知识库写入。")
        if tool_name in self._READ_TOOLS:
            return self._allow(f"使用 {tool_name}", "只读取现有信息。")
        if self._is_mcp_tool(tool_name):
            return self._evaluate_mcp_tool(tool_name)
        return self._ask(f"使用未知工具：{tool_name}", "这是未分类工具，需要先确认。")

    def audit(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        decision: PermissionDecision,
        temporary: bool = False,
        approval_id: str = "",
    ) -> None:
        """追加一条权限审计记录。"""
        try:
            ensure_dir(self.audit_file.parent)
            record = {
                "time": datetime.now().isoformat(),
                "session_key": session_key,
                "channel": channel,
                "chat_id": chat_id,
                "tool": tool_name,
                "arguments": self._argument_summary(arguments),
                "action": decision.action,
                "summary": decision.summary,
                "reason": decision.reason,
                "source": decision.source,
                "temporary": temporary,
                "approval_id": approval_id,
            }
            with open(self.audit_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            return

    def fingerprint(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """生成同类操作指纹。"""
        if tool_name == "exec":
            command = str(arguments.get("command") or "")
            first = self._split_command(command)[0] if command.strip() else ""
            return f"exec:{self._command_key(first)}"
        if tool_name in {"read_file", "write_file", "edit_file", "index_document"}:
            raw_path = arguments.get("path") or arguments.get("file_path") or ""
            return f"{tool_name}:{self._path_key(raw_path)}"
        if tool_name in {"manage_memory", "remember_user_info"}:
            return f"{tool_name}:{str(arguments.get('target') or 'user').lower()}"
        if tool_name == "cron":
            return f"cron:{str(arguments.get('action') or '').lower()}"
        if tool_name == "message":
            media = bool(arguments.get("media"))
            cross = bool(arguments.get("channel") or arguments.get("chat_id"))
            return f"message:media={media}:cross={cross}"
        return tool_name

    def _evaluate_config_rules(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> PermissionDecision | None:
        """匹配用户配置的 allow/ask/deny 规则。"""
        target = self._rule_target(tool_name, arguments)
        for action in ("deny", "ask", "allow"):
            for pattern in self.rules.get(action, []):
                if fnmatch.fnmatch(target, pattern):
                    return PermissionDecision(
                        action=action,
                        summary=f"使用 {tool_name}",
                        reason=f"匹配 permissions.json 规则：{pattern}",
                        source="config",
                    )
        return None

    def _evaluate_exec(self, arguments: dict[str, Any]) -> PermissionDecision:
        """逐段判断命令风险。"""
        command = str(arguments.get("command") or "")
        if contains_internal_url(command):
            return self._deny("执行命令", "命令包含内网或本机敏感地址。")

        for segment in self._split_command(command):
            for pattern, reason in self._DENY_COMMANDS:
                if pattern.search(segment):
                    return self._deny(f"执行命令：{self._short(segment)}", reason)
            for pattern, reason in self._ASK_COMMANDS:
                if pattern.search(segment):
                    return self._ask(f"执行命令：{self._short(segment)}", reason)
        return self._allow(f"执行命令：{self._short(command)}", "未匹配到需要确认的高风险命令。")

    def _evaluate_read_path(self, arguments: dict[str, Any]) -> PermissionDecision:
        """读取敏感文件或越界文件需要确认或禁止。"""
        resolved = self._resolve_path(arguments.get("path"))
        if resolved is None:
            return self._ask("读取文件", "文件路径无法可靠解析。")
        if not self._is_under_workspace(resolved):
            if self._is_sensitive_path(resolved):
                return self._deny(f"读取敏感工作区外文件：{resolved.name}", "不允许读取工作区外的敏感文件。")
            return self._ask(f"读取工作区外文件：{resolved}", "目标路径不在当前工作区内。")
        if self._is_sensitive_path(resolved):
            return self._ask(f"读取敏感文件：{resolved.name}", "目标文件可能包含配置、密钥或账号凭证。")
        return self._allow(f"读取文件：{resolved.name}", "普通工作区文件读取。")

    def _evaluate_write_path(self, tool_name: str, arguments: dict[str, Any]) -> PermissionDecision:
        """写入路径越界、敏感或长期工作记忆时需要确认。"""
        resolved = self._resolve_path(arguments.get("path"))
        if resolved is None:
            return self._ask("修改文件", "文件路径无法可靠解析。")
        if not self._is_under_workspace(resolved):
            return self._deny(f"修改工作区外文件：{resolved}", "不允许通过工具修改工作区外文件。")
        if self._is_user_memory_path(resolved):
            return self._allow("修改长期用户记忆", "USER.md 写入在当前设计中默认允许。")
        if self._is_working_memory_path(resolved):
            return self._ask("修改长期工作记忆文件", "会直接修改 memory/MEMORY.md，影响长期工作记忆。")
        if self._is_sensitive_path(resolved):
            return self._ask(f"修改敏感文件：{resolved.name}", "目标文件可能包含配置、密钥或账号凭证。")
        return self._allow(f"{'写入' if tool_name == 'write_file' else '编辑'}文件：{resolved.name}", "普通工作区文件修改。")

    def _evaluate_memory_update(self, arguments: dict[str, Any]) -> PermissionDecision:
        """长期工作记忆写入需要确认；USER.md 写入默认允许。"""
        target = str(arguments.get("target") or "user").lower()
        if target == "memory":
            return self._ask("写入长期工作记忆", "会把信息长期保存到 memory/MEMORY.md。")
        return self._allow("写入长期用户记忆", "USER.md 写入在当前设计中默认允许。")

    def _evaluate_cron(self, arguments: dict[str, Any]) -> PermissionDecision:
        """判断定时任务风险。"""
        action = str(arguments.get("action") or "").lower()
        if action in {"", "list"}:
            return self._allow("查看定时任务", "只读取定时任务状态。")
        if action == "add" and arguments.get("at") and not arguments.get("every_seconds") and not arguments.get("cron_expr"):
            return self._allow("设置一次性提醒", "一次性提醒默认允许。")
        if action == "add":
            return self._ask("设置定时任务", "会创建重复或长期定时任务。")
        if action == "remove":
            return self._ask("删除定时任务", "会删除已有定时任务。")
        return self._ask("变更定时任务", "未知定时任务动作需要确认。")

    def _evaluate_message(
        self,
        arguments: dict[str, Any],
        *,
        channel: str,
        chat_id: str,
    ) -> PermissionDecision:
        """判断消息发送是否会外发附件或跨会话。"""
        media = arguments.get("media") or []
        target_channel = str(arguments.get("channel") or channel)
        target_chat_id = str(arguments.get("chat_id") or chat_id)
        if media:
            return self._ask("发送带附件的消息", "会把本地文件发送到聊天渠道。")
        if target_channel != channel or target_chat_id != chat_id:
            return self._ask(f"跨会话发送消息到 {target_channel}:{target_chat_id}", "会向当前聊天之外的位置主动发送内容。")
        return self._allow("发送当前会话消息", "普通当前会话消息发送。")

    def _evaluate_web_fetch(self, arguments: dict[str, Any]) -> PermissionDecision:
        """判断网页读取是否访问内网地址。"""
        url = str(arguments.get("url") or "")
        if contains_internal_url(url):
            return self._deny("读取网页", "URL 指向内网或本机敏感地址。")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return self._deny("读取网页", "只允许 http 或 https URL。")
        return self._allow("读取网页", "普通外部网页读取。")

    def _evaluate_mcp_tool(self, tool_name: str) -> PermissionDecision:
        """保守处理 MCP 或外部工具。"""
        lower = tool_name.lower()
        if any(hint in lower for hint in self._WRITE_NAME_HINTS):
            return self._ask(f"使用外部工具：{tool_name}", "外部工具可能会写入、删除、发送或发布内容。")
        if any(hint in lower for hint in ("read", "list", "search", "get", "fetch")):
            return self._allow(f"使用外部只读工具：{tool_name}", "外部工具名称看起来是只读操作。")
        return self._ask(f"使用外部工具：{tool_name}", "未知外部工具需要确认。")

    def _load_rules(self) -> dict[str, list[str]]:
        """读取 workspace/permissions.json 中的简单通配规则。"""
        if not self.config_file.exists():
            return {"allow": [], "ask": [], "deny": []}
        try:
            data = json.loads(self.config_file.read_text(encoding="utf-8"))
        except Exception:
            return {"allow": [], "ask": [], "deny": []}
        rules: dict[str, list[str]] = {"allow": [], "ask": [], "deny": []}
        for action in rules:
            values = data.get(action, [])
            if isinstance(values, list):
                rules[action] = [
                    str(value)
                    for value in values
                    if isinstance(value, str) and not (action == "allow" and self._unsafe_rule(value))
                ]
        return rules

    def _rule_target(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """生成用户规则匹配目标。"""
        if tool_name == "exec":
            return f"exec({str(arguments.get('command') or '')})"
        if tool_name in {"read_file", "write_file", "edit_file"}:
            return f"{tool_name}({self._path_key(arguments.get('path') or '')})"
        if tool_name in {"manage_memory", "remember_user_info"}:
            return f"{tool_name}(target={str(arguments.get('target') or 'user').lower()})"
        return f"{tool_name}(*)"

    def _resolve_path(self, raw_path: Any) -> Path | None:
        """解析路径。"""
        if not raw_path:
            return None
        try:
            path = Path(str(raw_path)).expanduser()
            if not path.is_absolute():
                path = self.workspace / path
            return path.resolve(strict=False)
        except Exception:
            return None

    def _is_under_workspace(self, path: Path) -> bool:
        """判断路径是否在工作区内。"""
        return path == self.workspace or self.workspace in path.parents

    def _is_user_memory_path(self, path: Path) -> bool:
        """判断路径是否是根目录 USER.md。"""
        return path.name.lower() == "user.md" and path.parent == self.workspace

    def _is_working_memory_path(self, path: Path) -> bool:
        """判断路径是否是 memory/MEMORY.md。"""
        return (
            path.name.lower() == "memory.md"
            and path.parent.name.lower() == "memory"
            and path.parent.parent == self.workspace
        )

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

    def _is_mcp_tool(self, tool_name: str) -> bool:
        """判断是否像 MCP 或外部工具。"""
        return tool_name.startswith("mcp_") or tool_name.startswith("mcp__") or "__" in tool_name

    def _split_command(self, command: str) -> list[str]:
        """按常见 shell 控制符拆分命令片段。"""
        parts = re.split(r"\s*(?:&&|\|\||[;\n|])\s*", command)
        return [part.strip() for part in parts if part.strip()]

    def _command_key(self, command: str) -> str:
        """提取命令类别，用于临时授权同类操作。"""
        cleaned = " ".join(command.strip().split())
        if not cleaned:
            return ""
        lower = cleaned.lower()
        for prefix in ("git push", "git status", "git clean", "git reset --hard"):
            if lower.startswith(prefix):
                return prefix
        first = cleaned.split()[0]
        if first.lower() in {"remove-item", "rm", "del", "rmdir"}:
            return first.lower()
        return first.lower()

    def _path_key(self, raw_path: Any) -> str:
        """生成路径规则键。"""
        resolved = self._resolve_path(raw_path)
        if resolved is None:
            return str(raw_path or "")
        try:
            return resolved.relative_to(self.workspace).as_posix()
        except ValueError:
            return str(resolved)

    def _argument_summary(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """生成审计用参数摘要。"""
        summary: dict[str, Any] = {}
        for key, value in arguments.items():
            if key in {"content", "new_text", "old_text"}:
                text = str(value)
                summary[key] = self._short(text, 120)
            elif key in {"path", "file_path", "command", "query", "action", "target", "op"}:
                summary[key] = self._short(str(value), 160)
        return summary

    def _rag_summary(self, tool_name: str) -> str:
        """返回 RAG 删除类动作摘要。"""
        return {
            "clear_rag_store": "清空 RAG 知识库",
            "remove_rag_document": "删除 RAG 文档",
            "cleanup_expired_rag": "清理过期 RAG 文档",
        }.get(tool_name, f"使用 {tool_name}")

    def _rag_reason(self, tool_name: str) -> str:
        """返回 RAG 删除类动作原因。"""
        return {
            "clear_rag_store": "会删除所有已索引文档和向量数据。",
            "remove_rag_document": "会从知识库中移除指定文档数据。",
            "cleanup_expired_rag": "会删除过期文档的索引数据。",
        }.get(tool_name, "会修改知识库数据。")

    def _unsafe_rule(self, value: str) -> bool:
        """拒绝明显过宽的危险允许规则。"""
        compact = value.replace(" ", "").lower()
        return compact in {"exec(*)", "write_file(*)", "edit_file(*)", "read_file(*)"}

    def _allow(self, summary: str, reason: str, source: str = "default") -> PermissionDecision:
        """创建允许结果。"""
        return PermissionDecision("allow", summary, reason, source)

    def _ask(self, summary: str, reason: str, source: str = "default") -> PermissionDecision:
        """创建询问结果。"""
        return PermissionDecision("ask", summary, reason, source)

    def _deny(self, summary: str, reason: str, source: str = "default") -> PermissionDecision:
        """创建禁止结果。"""
        return PermissionDecision("deny", summary, reason, source)

    @staticmethod
    def _short(text: str, limit: int = 120) -> str:
        """缩短展示和审计文本。"""
        clean = " ".join(text.split())
        return clean if len(clean) <= limit else clean[:limit] + "..."


class TemporaryPermissionStore:
    """保存 gateway 当前运行期内的 30 分钟临时授权。"""

    def __init__(self, ttl_seconds: int = 1800):
        """初始化临时授权。"""
        self.ttl = timedelta(seconds=ttl_seconds)
        self._grants: list[TemporaryGrant] = []

    def add(self, session_key: str, fingerprint: str) -> None:
        """新增或刷新临时授权。"""
        self._cleanup()
        expires_at = datetime.now() + self.ttl
        self._grants = [
            grant
            for grant in self._grants
            if not (grant.session_key == session_key and grant.fingerprint == fingerprint)
        ]
        self._grants.append(TemporaryGrant(session_key, fingerprint, expires_at))

    def allows(self, session_key: str, fingerprint: str) -> bool:
        """判断临时授权是否有效。"""
        self._cleanup()
        return any(
            grant.session_key == session_key and grant.fingerprint == fingerprint
            for grant in self._grants
        )

    def _cleanup(self) -> None:
        """清理过期授权。"""
        now = datetime.now()
        self._grants = [grant for grant in self._grants if grant.expires_at > now]
