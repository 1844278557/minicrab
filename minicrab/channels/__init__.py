"""Chat channels module with plugin architecture."""

from minicrab.channels.base import BaseChannel
from minicrab.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
