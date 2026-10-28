"""测试统一工具权限层。"""

from __future__ import annotations

import json
from pathlib import Path

from minicrab.agent.permissions import PermissionDecision, PermissionPolicy, TemporaryPermissionStore


def test_permission_policy_allows_common_read_tools(tmp_path: Path) -> None:
    """普通只读工具默认允许。"""
    policy = PermissionPolicy(tmp_path)

    assert policy.evaluate("read_file", {"path": "notes.md"}).action == "allow"
    assert policy.evaluate("list_dir", {"path": "."}).action == "allow"
    assert policy.evaluate("web_search", {"query": "minicrab"}).action == "allow"


def test_permission_policy_asks_for_sensitive_read(tmp_path: Path) -> None:
    """读取敏感文件需要确认。"""
    policy = PermissionPolicy(tmp_path)

    decision = policy.evaluate("read_file", {"path": ".env"})

    assert decision.action == "ask"
    assert "敏感文件" in decision.summary


def test_permission_policy_denies_sensitive_outside_read(tmp_path: Path) -> None:
    """工作区外敏感文件默认禁止读取。"""
    policy = PermissionPolicy(tmp_path)

    decision = policy.evaluate("read_file", {"path": str(tmp_path.parent / ".env")})

    assert decision.action == "deny"


def test_permission_policy_splits_shell_segments(tmp_path: Path) -> None:
    """复合命令中任一片段有风险都要识别。"""
    policy = PermissionPolicy(tmp_path)

    decision = policy.evaluate("exec", {"command": "git status; git push"})

    assert decision.action == "ask"
    assert "git push" in decision.summary


def test_permission_policy_denies_destructive_shell(tmp_path: Path) -> None:
    """明确破坏性命令默认禁止。"""
    policy = PermissionPolicy(tmp_path)

    decision = policy.evaluate("exec", {"command": "format C:"})

    assert decision.action == "deny"


def test_permission_policy_handles_mcp_tools(tmp_path: Path) -> None:
    """MCP 外部工具按只读和写入风险分类。"""
    policy = PermissionPolicy(tmp_path)

    assert policy.evaluate("mcp__drive__read_file", {}).action == "allow"
    assert policy.evaluate("mcp__drive__delete_file", {}).action == "ask"
    assert policy.evaluate("mcp__unknown__run", {}).action == "ask"


def test_permission_policy_config_rules_override_defaults(tmp_path: Path) -> None:
    """permissions.json 可以用通配规则覆盖默认行为。"""
    (tmp_path / "permissions.json").write_text(
        json.dumps({"deny": ["exec(git status*)"]}),
        encoding="utf-8",
    )
    policy = PermissionPolicy(tmp_path)

    decision = policy.evaluate("exec", {"command": "git status"})

    assert decision.action == "deny"
    assert decision.source == "config"


def test_permission_audit_writes_jsonl(tmp_path: Path) -> None:
    """权限判断应写入可排查的审计日志。"""
    policy = PermissionPolicy(tmp_path)

    policy.audit(
        session_key="web:default",
        channel="web",
        chat_id="default",
        tool_name="exec",
        arguments={"command": "git status"},
        decision=PermissionDecision("allow", "执行命令", "普通查看命令。"),
    )

    lines = (tmp_path / "memory" / "permissions.log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["action"] == "allow"
    assert record["tool"] == "exec"


def test_temporary_permission_store_expires_by_fingerprint() -> None:
    """临时授权按会话和指纹生效。"""
    store = TemporaryPermissionStore(ttl_seconds=1800)

    store.add("web:default", "exec:git push")

    assert store.allows("web:default", "exec:git push") is True
    assert store.allows("weixin:user", "exec:git push") is False
