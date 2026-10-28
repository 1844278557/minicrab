"""Slash command routing and built-in handlers."""

from minicrab.command.builtin import register_builtin_commands
from minicrab.command.router import CommandContext, CommandRouter

__all__ = ["CommandContext", "CommandRouter", "register_builtin_commands"]
