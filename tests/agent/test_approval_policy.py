"""测试人工审核策略只拦截明确高风险动作。"""

from __future__ import annotations

from minicrab.agent.approval import ApprovalPolicy


def test_remember_user_info_user_target_does_not_require_approval(tmp_path) -> None:
    """USER.md 写入不再需要审核，MEMORY.md 显式写入仍需要审核。"""
    policy = ApprovalPolicy(tmp_path)

    decision = policy.evaluate("remember_user_info", {"content": "生日是 5.20"})

    assert decision is None
    manage_user_decision = policy.evaluate(
        "manage_memory",
        {"target": "user", "op": "append", "content": "- Prefer concise replies."},
    )
    assert manage_user_decision is None
    memory_decision = policy.evaluate(
        "remember_user_info",
        {"content": "审批弹窗不进入历史消息", "target": "memory"},
    )
    assert memory_decision is not None
    assert "MEMORY.md" in memory_decision.reason
    manage_memory_decision = policy.evaluate(
        "manage_memory",
        {"target": "memory", "op": "append", "content": "- [decision][2026-06-04] Web approval uses modal."},
    )
    assert manage_memory_decision is not None
    assert "MEMORY.md" in manage_memory_decision.reason


def test_safe_exec_does_not_require_approval(tmp_path) -> None:
    """普通查看类命令不需要审核。"""
    policy = ApprovalPolicy(tmp_path)

    assert policy.evaluate("exec", {"command": "git status"}) is None
    assert policy.evaluate("exec", {"command": "pytest tests -q"}) is None


def test_high_risk_exec_requires_approval(tmp_path) -> None:
    """推送、删除、安装类命令需要审核。"""
    policy = ApprovalPolicy(tmp_path)

    assert policy.evaluate("exec", {"command": "git push"}) is not None
    assert policy.evaluate("exec", {"command": "Remove-Item .pytest_cache -Recurse"}) is not None
    assert policy.evaluate("exec", {"command": "pip install requests"}) is not None


def test_workspace_write_policy(tmp_path) -> None:
    """工作区内普通文件不审核，敏感文件和工作区外路径要审核。"""
    policy = ApprovalPolicy(tmp_path)

    assert policy.evaluate("write_file", {"path": "notes.md", "content": "ok"}) is None
    assert policy.evaluate("edit_file", {"path": "USER.md", "old_text": "a", "new_text": "b"}) is None
    assert policy.evaluate("edit_file", {"path": "memory/MEMORY.md", "old_text": "a", "new_text": "b"}) is not None
    assert policy.evaluate("edit_file", {"path": ".env", "old_text": "a", "new_text": "b"}) is not None
    assert policy.evaluate("write_file", {"path": str(tmp_path.parent / "outside.txt"), "content": "x"}) is not None


def test_sensitive_read_policy(tmp_path) -> None:
    """普通读取不审核，敏感文件读取要进入权限层。"""
    policy = ApprovalPolicy(tmp_path)

    assert policy.evaluate("read_file", {"path": "notes.md"}) is None
    assert policy.evaluate("read_file", {"path": ".env"}) is not None


def test_cron_and_rag_delete_actions_require_approval(tmp_path) -> None:
    """重复定时任务和清空删除类 RAG 工具需要审核。"""
    policy = ApprovalPolicy(tmp_path)

    assert policy.evaluate("cron", {"action": "list"}) is None
    assert policy.evaluate(
        "cron",
        {"action": "add", "message": "提醒我", "at": "2026-05-20T19:46:00"},
    ) is None
    assert policy.evaluate(
        "cron",
        {"action": "add", "message": "提醒我", "every_seconds": 300},
    ) is not None
    assert policy.evaluate("cron", {"action": "remove", "job_id": "1"}) is not None
    assert policy.evaluate("clear_rag_store", {"confirm": "yes"}) is not None
    assert policy.evaluate("remove_rag_document", {"file_path": "a.md"}) is not None


def test_message_requires_approval_only_for_external_or_media_send(tmp_path) -> None:
    """当前会话普通消息不审核，附件或跨会话外发要审核。"""
    policy = ApprovalPolicy(tmp_path)

    assert policy.evaluate(
        "message",
        {"content": "hello"},
        channel="web",
        chat_id="default",
    ) is None
    assert policy.evaluate(
        "message",
        {"content": "hello", "media": ["report.pdf"]},
        channel="web",
        chat_id="default",
    ) is not None
    assert policy.evaluate(
        "message",
        {"content": "hello", "channel": "weixin", "chat_id": "other"},
        channel="web",
        chat_id="default",
    ) is not None


def test_unknown_and_mcp_tools_are_classified(tmp_path) -> None:
    """未知工具默认确认，MCP 只读工具可放行，写入类工具要确认。"""
    policy = ApprovalPolicy(tmp_path)

    assert policy.evaluate("unknown_tool", {}) is not None
    assert policy.evaluate("mcp__drive__read_file", {}) is None
    assert policy.evaluate("mcp__drive__delete_file", {}) is not None
