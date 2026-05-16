"""Web 渠道的基础行为测试。"""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.channels.web import WebChannel, WebConfig
from nanobot.config.schema import ChannelsConfig


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

    with patch("nanobot.channels.registry.discover_all", return_value={"web": WebChannel}):
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
    assert "Nanobot Web" in response.text


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


@pytest.mark.asyncio
async def test_send_pushes_outbound_message_to_browser_socket() -> None:
    """Agent 回复会被推送给已连接浏览器。"""
    channel = WebChannel({"enabled": True, "allowFrom": ["*"]}, MessageBus())
    socket = _FakeSocket()
    channel._register("default", socket)  # type: ignore[arg-type]

    await channel.send(OutboundMessage(channel="web", chat_id="default", content="收到"))

    assert socket.sent == [{"type": "message", "role": "assistant", "content": "收到"}]


@pytest.mark.asyncio
async def test_send_without_browser_does_not_raise() -> None:
    """浏览器未连接时，发送回复不会导致异常。"""
    channel = WebChannel({"enabled": True, "allowFrom": ["*"]}, MessageBus())

    await channel.send(OutboundMessage(channel="web", chat_id="missing", content="收到"))

    assert channel._connections == {}


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, str]] = []

    async def send_json(self, payload: dict[str, str]) -> None:
        self.sent.append(payload)
