"""Web 聊天渠道，用浏览器作为 Nanobot 的本地交流入口。"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base


class WebConfig(Base):
    """Web 渠道配置。"""

    enabled: bool = False
    allow_from: list[str] = Field(default_factory=lambda: ["*"])
    host: str = "127.0.0.1"
    port: int = 18790
    path: str = "/web"
    ws_path: str = "/web/ws"
    streaming: bool = False


class WebChannel(BaseChannel):
    """通过本地 Web 页面收发消息的渠道。"""

    name = "web"
    display_name = "Web"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WebConfig().model_dump(by_alias=True)

    def __init__(
        self,
        config: Any,
        bus: MessageBus,
        history_provider: Callable[[str, int], Any] | None = None,
    ):
        if isinstance(config, dict):
            config = WebConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: WebConfig = config
        self._history_provider = history_provider
        self._connections: dict[str, list[WebSocket]] = {}
        self._server: uvicorn.Server | None = None
        self.app = self._create_app()

    def set_history_provider(self, provider: Callable[[str, int], Any] | None) -> None:
        """设置 Web 历史读取回调。"""
        self._history_provider = provider

    async def start(self) -> None:
        """启动 Web 渠道。"""
        self._running = True
        server_config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            access_log=False,
            log_level="info",
        )
        self._server = uvicorn.Server(server_config)
        logger.info("Web channel serving at http://{}{}", self._address, self.config.path)
        try:
            await self._server.serve()
        finally:
            self._running = False

    async def stop(self) -> None:
        """停止 Web 渠道。"""
        self._running = False
        if self._server:
            self._server.should_exit = True
        for chat_id, sockets in list(self._connections.items()):
            for socket in list(sockets):
                try:
                    await socket.close(code=1001)
                except Exception:
                    logger.debug("Web channel: websocket already closed for {}", chat_id)
            self._connections.pop(chat_id, None)

    async def send(self, msg: OutboundMessage) -> None:
        """发送 Agent 回复。"""
        sockets = list(self._connections.get(msg.chat_id, []))
        if not sockets:
            logger.debug("Web channel: no active browser for chat {}", msg.chat_id)
            return

        payload = self._outbound_payload(msg)
        stale: list[WebSocket] = []
        for socket in sockets:
            try:
                await socket.send_json(payload)
            except Exception as exc:
                logger.warning("Web channel send failed for {}: {}", msg.chat_id, exc)
                stale.append(socket)
        for socket in stale:
            self._unregister(msg.chat_id, socket)

    @property
    def _address(self) -> str:
        """返回用于日志展示的监听地址。"""
        return f"{self.config.host}:{self.config.port}"

    @property
    def _template_dir(self) -> Path:
        """返回 Web 页面模板目录。"""
        return Path(__file__).resolve().parents[1] / "templates" / "web"

    def _create_app(self) -> FastAPI:
        """创建 Web 渠道使用的 FastAPI 应用。"""
        app = FastAPI(title="Nanobot Web")
        template_dir = self._template_dir
        app.mount(
            f"{self.config.path}/assets",
            StaticFiles(directory=template_dir),
            name="web_assets",
        )

        @app.get(self.config.path)
        @app.get(f"{self.config.path}/")
        async def index() -> FileResponse:
            return FileResponse(template_dir / "index.html")

        @app.websocket(self.config.ws_path)
        async def websocket_endpoint(socket: WebSocket) -> None:
            await self._handle_websocket(socket)

        return app

    async def _handle_websocket(self, socket: WebSocket) -> None:
        """处理一个浏览器 WebSocket 连接。"""
        chat_id = str(socket.query_params.get("chat_id") or "default")
        sender_id = str(socket.query_params.get("sender_id") or "web-user")
        if not self.is_allowed(sender_id):
            await socket.close(code=1008)
            return

        await socket.accept()
        self._register(chat_id, socket)
        await socket.send_json({"type": "status", "content": "connected"})
        try:
            async for payload in socket.iter_json():
                await self._handle_client_payload(payload, socket, chat_id, sender_id)
        except WebSocketDisconnect:
            logger.debug("Web channel websocket disconnected: {}", chat_id)
        finally:
            self._unregister(chat_id, socket)

    async def _handle_client_payload(
        self,
        payload: Any,
        socket: WebSocket,
        chat_id: str,
        sender_id: str,
    ) -> None:
        """处理浏览器发来的单条消息。"""
        if not isinstance(payload, dict):
            await socket.send_json({"type": "error", "content": "不支持的消息格式。"})
            return

        payload_type = payload.get("type", "message")
        if payload_type == "history":
            await self._send_history(socket, chat_id, payload)
            return
        if payload_type != "message":
            await socket.send_json({"type": "error", "content": "不支持的消息格式。"})
            return

        content = str(payload.get("content") or "").strip()
        if not content:
            await socket.send_json({"type": "error", "content": "消息不能为空。"})
            return

        await socket.send_json({"type": "status", "content": "processing"})
        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            metadata={"source": "web"},
        )

    async def _send_history(self, socket: WebSocket, chat_id: str, payload: dict[str, Any]) -> None:
        """向浏览器返回当前 Web 会话最近历史。"""
        limit = self._history_limit(payload.get("limit", 100))
        messages: list[dict[str, str]] = []
        if self._history_provider:
            session_key = f"{self.name}:{chat_id}"
            result = self._history_provider(session_key, limit)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, list):
                messages = result[-limit:] if limit else []
        await socket.send_json({"type": "history", "messages": messages})

    @staticmethod
    def _history_limit(value: Any) -> int:
        """限制 Web 历史最多返回 100 条。"""
        try:
            limit = int(value)
        except (TypeError, ValueError):
            limit = 100
        return max(0, min(limit, 100))

    def _register(self, chat_id: str, socket: WebSocket) -> None:
        """登记一个浏览器连接。"""
        sockets = self._connections.setdefault(chat_id, [])
        if socket not in sockets:
            sockets.append(socket)

    def _unregister(self, chat_id: str, socket: WebSocket) -> None:
        """移除一个浏览器连接。"""
        sockets = self._connections.get(chat_id)
        if not sockets:
            return
        if socket in sockets:
            sockets.remove(socket)
        if not sockets:
            self._connections.pop(chat_id, None)

    @staticmethod
    def _outbound_payload(msg: OutboundMessage) -> dict[str, Any]:
        """把 Agent 回复转成浏览器消息格式。"""
        if msg.metadata.get("_progress"):
            return {"type": "status", "content": msg.content or "processing"}
        payload: dict[str, Any] = {"type": "message", "role": "assistant", "content": msg.content}
        approval = msg.metadata.get("approval")
        if isinstance(approval, dict):
            payload["approval"] = approval
        return payload
