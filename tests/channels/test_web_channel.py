"""Web 渠道的基础行为测试。"""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from minicrab.bus.events import OutboundMessage
from minicrab.bus.queue import MessageBus
from minicrab.channels.manager import ChannelManager
from minicrab.channels.web import WebChannel, WebConfig
from minicrab.config.schema import ChannelsConfig


def test_web_channel_default_config_is_disabled() -> None:
    """默认配置不主动启用 Web 渠道。"""
    cfg = WebChannel.default_config()

    assert cfg["enabled"] is False
    assert cfg["host"] == "127.0.0.1"
    assert cfg["allowFrom"] == ["*"]


def test_web_channel_init_from_dict() -> None:
    """Web 渠道可以从配置字典初始化。"""
    channel = WebChannel(
        {"enabled": True, "allowFrom": ["*"], "host": "127.0.0.1", "port": 18790},
        MessageBus(),
    )

    assert isinstance(channel.config, WebConfig)
    assert channel.config.enabled is True
    assert channel.config.allow_from == ["*"]


def test_channel_manager_loads_web_channel_from_config() -> None:
    """配置启用 web 后，ChannelManager 能加载 WebChannel。"""
    fake_config = SimpleNamespace(
        channels=ChannelsConfig.model_validate({
            "web": {"enabled": True, "allowFrom": ["*"]},
        }),
        providers=SimpleNamespace(groq=SimpleNamespace(api_key="")),
    )

    with patch("minicrab.channels.registry.discover_all", return_value={"web": WebChannel}):
        manager = ChannelManager.__new__(ChannelManager)
        manager.config = fake_config
        manager.bus = MessageBus()
        manager.channels = {}
        manager._dispatch_task = None
        manager._init_channels()

    assert "web" in manager.channels
    assert isinstance(manager.channels["web"], WebChannel)


def test_web_page_is_served() -> None:
    """Web 渠道提供聊天页面。"""
    channel = WebChannel({"enabled": True, "allowFrom": ["*"]}, MessageBus())
    client = TestClient(channel.app)

    response = client.get("/web")

    assert response.status_code == 200
    assert "minicrab" in response.text
    assert "/web/assets/minicrab-icon.png" in response.text
    icon = client.get("/web/assets/minicrab-icon.png")
    assert icon.status_code == 200
    assert icon.headers["content-type"] == "image/png"
    assert icon.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_web_assets_include_markdown_rendering() -> None:
    """Web 前端资源包含本地 Markdown 格式化渲染逻辑。"""
    channel = WebChannel({"enabled": True, "allowFrom": ["*"]}, MessageBus())
    client = TestClient(channel.app)

    script = client.get("/web/assets/app.js")
    style = client.get("/web/assets/style.css")

    assert script.status_code == 200
    assert "function renderMarkdown" in script.text
    assert "function escapeHtml" in script.text
    assert "text.innerHTML = renderMarkdown(content)" in script.text
    assert style.status_code == 200
    assert ".message-content pre" in style.text


def test_websocket_message_publishes_inbound_message() -> None:
    """浏览器消息会进入 MessageBus。"""
    bus = MessageBus()
    channel = WebChannel({"enabled": True, "allowFrom": ["*"]}, bus)
    client = TestClient(channel.app)

    with client.websocket_connect("/web/ws") as socket:
        assert socket.receive_json() == {"type": "status", "content": "connected"}
        socket.send_json({"type": "message", "content": "你好"})
        assert socket.receive_json() == {"type": "status", "content": "processing"}

        inbound = None
        for _ in range(20):
            try:
                inbound = bus.inbound.get_nowait()
                break
            except Exception:
                time.sleep(0.01)

    assert inbound is not None
    assert inbound.channel == "web"
    assert inbound.chat_id == "default"
    assert inbound.sender_id == "web-user"
    assert inbound.content == "你好"
    assert inbound.session_key == "web:default"


def test_websocket_approval_response_publishes_text_confirmation() -> None:
    """浏览器审批弹窗会转成渠道文本确认。"""
    bus = MessageBus()
    channel = WebChannel({"enabled": True, "allowFrom": ["*"]}, bus)
    client = TestClient(channel.app)

    with client.websocket_connect("/web/ws") as socket:
        assert socket.receive_json() == {"type": "status", "content": "connected"}
        socket.send_json({"type": "approval_response", "decision": "approved", "id": "A-test"})
        assert socket.receive_json() == {"type": "status", "content": "processing"}

        inbound = None
        for _ in range(20):
            try:
                inbound = bus.inbound.get_nowait()
                break
            except Exception:
                time.sleep(0.01)

    assert inbound is not None
    assert inbound.channel == "web"
    assert inbound.content == "同意 A-test"
    assert inbound.metadata["_approval_response"] is True


def test_websocket_temporary_approval_response_publishes_text_confirmation() -> None:
    """浏览器 30 分钟允许会转成文字确认。"""
    bus = MessageBus()
    channel = WebChannel({"enabled": True, "allowFrom": ["*"]}, bus)
    client = TestClient(channel.app)

    with client.websocket_connect("/web/ws") as socket:
        assert socket.receive_json() == {"type": "status", "content": "connected"}
        socket.send_json({"type": "approval_response", "decision": "approved_temporary", "id": "A-test"})
        assert socket.receive_json() == {"type": "status", "content": "processing"}

        inbound = None
        for _ in range(20):
            try:
                inbound = bus.inbound.get_nowait()
                break
            except Exception:
                time.sleep(0.01)

    assert inbound is not None
    assert inbound.channel == "web"
    assert inbound.content == "确认30分钟 A-test"
    assert inbound.metadata["_approval_response"] is True


def test_websocket_history_without_provider_returns_empty_list() -> None:
    """没有历史读取回调时，Web 历史请求返回空列表。"""
    channel = WebChannel({"enabled": True, "allowFrom": ["*"]}, MessageBus())
    client = TestClient(channel.app)

    with client.websocket_connect("/web/ws") as socket:
        assert socket.receive_json() == {"type": "status", "content": "connected"}
        socket.send_json({"type": "history", "limit": 100})

        assert socket.receive_json() == {"type": "history", "messages": []}


def test_websocket_history_uses_provider_and_caps_limit() -> None:
    """Web 历史请求会调用 provider，并把 limit 限制到 100。"""
    calls: list[tuple[str, int]] = []

    def history_provider(session_key: str, limit: int) -> list[dict[str, str]]:
        calls.append((session_key, limit))
        return [{"role": "user", "content": f"m{i}", "timestamp": ""} for i in range(120)]

    channel = WebChannel(
        {"enabled": True, "allowFrom": ["*"]},
        MessageBus(),
        history_provider=history_provider,
    )
    client = TestClient(channel.app)

    with client.websocket_connect("/web/ws?chat_id=custom") as socket:
        assert socket.receive_json() == {"type": "status", "content": "connected"}
        socket.send_json({"type": "history", "limit": 1000})
        response = socket.receive_json()

    assert calls == [("web:custom", 100)]
    assert response["type"] == "history"
    assert len(response["messages"]) == 100
    assert response["messages"][0]["content"] == "m20"


def test_websocket_history_zero_limit_returns_empty_list() -> None:
    """limit 为 0 时不会返回全部历史。"""
    channel = WebChannel(
        {"enabled": True, "allowFrom": ["*"]},
        MessageBus(),
        history_provider=lambda _session_key, _limit: [
            {"role": "user", "content": "hello", "timestamp": ""}
        ],
    )
    client = TestClient(channel.app)

    with client.websocket_connect("/web/ws") as socket:
        assert socket.receive_json() == {"type": "status", "content": "connected"}
        socket.send_json({"type": "history", "limit": 0})

        assert socket.receive_json() == {"type": "history", "messages": []}


@pytest.mark.asyncio
async def test_send_pushes_outbound_message_to_browser_socket() -> None:
    """Agent 回复会被推送给已连接浏览器。"""
    channel = WebChannel({"enabled": True, "allowFrom": ["*"]}, MessageBus())
    socket = _FakeSocket()
    channel._register("default", socket)  # type: ignore[arg-type]

    await channel.send(OutboundMessage(channel="web", chat_id="default", content="收到"))

    assert socket.sent == [{"type": "message", "role": "assistant", "content": "收到"}]


@pytest.mark.asyncio
async def test_send_pushes_approval_metadata_to_browser_socket() -> None:
    """审批控制事件会把弹窗所需元数据发给浏览器。"""
    channel = WebChannel({"enabled": True, "allowFrom": ["*"]}, MessageBus())
    socket = _FakeSocket()
    channel._register("default", socket)  # type: ignore[arg-type]

    await channel.send(OutboundMessage(
        channel="web",
        chat_id="default",
        content="",
        metadata={
            "_control": "approval_required",
            "approval": {
                "id": "A-test",
                "summary": "写入长期用户记忆",
                "reason": "会写入 USER.md",
            }
        },
    ))

    assert socket.sent == [{
        "type": "approval_required",
        "approval": {
            "id": "A-test",
            "summary": "写入长期用户记忆",
            "reason": "会写入 USER.md",
        },
    }]


@pytest.mark.asyncio
async def test_send_without_browser_does_not_raise() -> None:
    """浏览器未连接时，发送回复不会导致异常。"""
    channel = WebChannel({"enabled": True, "allowFrom": ["*"]}, MessageBus())

    await channel.send(OutboundMessage(channel="web", chat_id="missing", content="收到"))

    assert channel._connections == {}


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)
