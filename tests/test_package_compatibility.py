"""验证新包名与旧插件、环境变量、技能元数据的兼容边界。"""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

from minicrab.agent.skills import SkillsLoader
from minicrab.channels.base import BaseChannel
from minicrab.channels.registry import discover_plugins
from minicrab.config.schema import Config


def test_legacy_imports_share_module_and_class_identity():
    """旧插件和新代码必须使用同一个基类，而不是加载两份模块。"""
    for suffix in ("channels.base", "bus.events", "config.loader", "agent.runner"):
        current = importlib.import_module(f"minicrab.{suffix}")
        legacy = importlib.import_module(f"nanobot.{suffix}")
        assert legacy is current
        assert current.__spec__.name == f"minicrab.{suffix}"

    assert importlib.import_module("nanobot.channels.base").BaseChannel is BaseChannel


def test_new_environment_prefix_takes_priority(monkeypatch):
    """已有旧环境变量时，新前缀可以覆盖同一配置。"""
    monkeypatch.setenv("NANOBOT_AGENTS__DEFAULTS__MODEL", "old-model")
    monkeypatch.setenv("MINICRAB_AGENTS__DEFAULTS__MODEL", "new-model")
    assert Config().agents.defaults.model == "new-model"


def test_old_environment_prefix_still_loads(monkeypatch):
    """未设置新环境变量时，原来的启动配置仍然有效。"""
    monkeypatch.delenv("MINICRAB_AGENTS__DEFAULTS__MODEL", raising=False)
    monkeypatch.setenv("NANOBOT_AGENTS__DEFAULTS__MODEL", "old-model")
    assert Config().agents.defaults.model == "old-model"


def test_plugin_groups_preserve_old_plugins_and_prefer_new_names():
    """加载旧入口中的插件，同名时以新入口为准。"""
    current = SimpleNamespace(name="shared", load=lambda: "new-plugin")
    legacy = SimpleNamespace(name="shared", load=lambda: "old-plugin")
    legacy_only = SimpleNamespace(name="legacy", load=lambda: "legacy-plugin")

    def entries(*, group):
        """模拟新旧入口同时安装。"""
        return [current] if group == "minicrab.channels" else [legacy, legacy_only]

    with patch("importlib.metadata.entry_points", side_effect=entries):
        assert discover_plugins() == {"shared": "new-plugin", "legacy": "legacy-plugin"}


def test_skill_metadata_accepts_new_and_old_names(tmp_path):
    """已有技能元数据仍可读取，新名称可覆盖旧内容。"""
    loader = SkillsLoader(tmp_path)
    assert loader._parse_minicrab_metadata('{"nanobot":{"always":true}}') == {"always": True}
    assert loader._parse_minicrab_metadata('{"minicrab":{"always":false},"nanobot":{"always":true}}') == {"always": False}
