"""Message bus module for decoupled channel-agent communication."""

from minicrab.bus.events import InboundMessage, OutboundMessage
from minicrab.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
