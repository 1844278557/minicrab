"""测试人工审核策略只拦截明确高风险动作。"""

from __future__ import annotations

from nanobot.agent.approval import ApprovalPolicy


def test_remember_user_info_requires_approval(tmp_path) -> None:
    """写入长期用户记忆需要审核。"""
    policy = ApprovalPolicy(tmp_path)

    decision = policy.evaluate("remember_user_info", {"content": "生日是 5.20"})

    assert decision is not None
    assert "USER.md" in decision.reason


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
    assert policy.evaluate("edit_file", {"path": "USER.md", "old_text": "a", "new_text": "b"}) is not None
    assert policy.evaluate("edit_file", {"path": ".env", "old_text": "a", "new_text": "b"}) is not None
    assert policy.evaluate("write_file", {"path": str(tmp_path.parent / "outside.txt"), "content": "x"}) is not None


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
