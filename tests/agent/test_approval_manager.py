"""测试人工确认文案和用户回复解析。"""

from __future__ import annotations

from nanobot.agent.approval import ApprovalManager


def test_single_approval_prompt_hides_internal_id(tmp_path) -> None:
    """单个确认事项不向用户展示内部编号。"""
    manager = ApprovalManager(tmp_path)

    event = manager.request(
        session_key="weixin:user",
        channel="weixin",
        chat_id="user",
        tool_name="remember_user_info",
        arguments={"content": "# User Profile\n\n- 使用 Python。"},
    )

    assert event is not None
    content = event["content"]
    assert "要我更新长期记忆吗？" in content
    assert "回复“确认”继续" in content
    assert "A-" not in content
    assert "编号" not in content
    assert "remember_user_info" not in content


def test_plain_confirm_consumes_single_pending_request(tmp_path) -> None:
    """只有一个待确认事项时，用户回复确认即可。"""
    manager = ApprovalManager(tmp_path)
    event = manager.request(
        session_key="weixin:user",
        channel="weixin",
        chat_id="user",
        tool_name="clear_rag_store",
        arguments={"confirm": "yes"},
    )

    resolution = manager.consume_response("weixin:user", "确认")

    assert event is not None
    assert resolution is not None
    assert resolution.status == "approved"
    assert resolution.request is not None
    assert resolution.request.id == event["approval_id"]


def test_multiple_approval_prompt_uses_numbered_choices(tmp_path) -> None:
    """多个待确认事项时，用序号确认，不展示内部编号。"""
    manager = ApprovalManager(tmp_path)
    first = manager.request(
        session_key="weixin:user",
        channel="weixin",
        chat_id="user",
        tool_name="remember_user_info",
        arguments={"content": "# User Profile\n\n- 使用 Python。"},
    )
    second = manager.request(
        session_key="weixin:user",
        channel="weixin",
        chat_id="user",
        tool_name="clear_rag_store",
        arguments={"confirm": "yes"},
    )

    assert first is not None
    assert second is not None
    assert "1. 更新长期记忆" in second["content"]
    assert "2. 清空知识库" in second["content"]
    assert "确认 1" in second["content"]
    assert "A-" not in second["content"]

    resolution = manager.consume_response("weixin:user", "确认 1")

    assert resolution is not None
    assert resolution.status == "approved"
    assert resolution.request is not None
    assert resolution.request.id == first["approval_id"]


def test_legacy_confirm_with_internal_id_still_works(tmp_path) -> None:
    """旧的 同意 A-xxx 回复仍然兼容。"""
    manager = ApprovalManager(tmp_path)
    event = manager.request(
        session_key="weixin:user",
        channel="weixin",
        chat_id="user",
        tool_name="clear_rag_store",
        arguments={"confirm": "yes"},
    )

    assert event is not None
    resolution = manager.consume_response("weixin:user", f"同意 {event['approval_id']}")

    assert resolution is not None
    assert resolution.status == "approved"
