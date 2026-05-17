"""测试用户主动记忆写入 USER.md 的行为。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.memory import RememberUserInfoTool
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings


@pytest.mark.asyncio
async def test_remember_user_info_creates_user_md(tmp_path) -> None:
    """用户主动记忆工具会创建 USER.md。"""
    tool = RememberUserInfoTool(workspace=tmp_path)
    content = "# User Profile\n\n- 女朋友生日是 5 月 20 日。"

    result = await tool.execute(content=content, reason="用户明确要求记住")

    assert result.startswith("USER.md updated.")
    assert (tmp_path / "USER.md").read_text(encoding="utf-8") == content + "\n"


@pytest.mark.asyncio
async def test_remember_user_info_replaces_existing_user_md(tmp_path) -> None:
    """用户主动记忆工具会按完整内容更新 USER.md。"""
    user_file = tmp_path / "USER.md"
    user_file.write_text("# User Profile\n\n- 喜欢早上提醒。", encoding="utf-8")
    tool = RememberUserInfoTool(workspace=tmp_path)
    content = "# User Profile\n\n- 喜欢早上提醒。\n- 女朋友生日是 5 月 20 日。"

    await tool.execute(content=content)

    assert user_file.read_text(encoding="utf-8") == content + "\n"


@pytest.mark.asyncio
async def test_remember_user_info_rejects_empty_content(tmp_path) -> None:
    """空内容不会覆盖已有 USER.md。"""
    user_file = tmp_path / "USER.md"
    user_file.write_text("# User Profile\n\n- 已有信息。", encoding="utf-8")
    tool = RememberUserInfoTool(workspace=tmp_path)

    result = await tool.execute(content="   ")

    assert result.startswith("Error:")
    assert user_file.read_text(encoding="utf-8") == "# User Profile\n\n- 已有信息。"


def test_user_md_is_loaded_into_system_prompt(tmp_path) -> None:
    """USER.md 内容会进入系统提示。"""
    (tmp_path / "USER.md").write_text("# User Profile\n\n- 女朋友生日是 5 月 20 日。", encoding="utf-8")

    prompt = ContextBuilder(tmp_path).build_system_prompt()

    assert "## USER.md" in prompt
    assert "女朋友生日是 5 月 20 日" in prompt


def test_agent_loop_registers_remember_user_info_tool(tmp_path) -> None:
    """AgentLoop 默认注册用户主动记忆工具。"""
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.estimate_prompt_tokens.return_value = (0, "test-counter")

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )

    assert loop.tools.has("remember_user_info")
    schema = loop.tools.get("remember_user_info").to_schema()
    assert schema["function"]["parameters"]["required"] == ["content"]
