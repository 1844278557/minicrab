"""验证 minicrab 的默认目录、指定配置和工作区解析。"""

from pathlib import Path

import pytest

from minicrab.config.loader import get_config_path
from minicrab.config.schema import Config

from minicrab.config.paths import (
    get_bridge_install_dir,
    get_cli_history_path,
    get_cron_dir,
    get_data_dir,
    get_legacy_sessions_dir,
    get_logs_dir,
    get_media_dir,
    get_runtime_subdir,
    get_workspace_path,
    is_default_workspace,
)


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    """把默认目录检测限制在测试目录内，不依赖实际用户配置。"""
    original_expanduser = Path.expanduser

    def expanduser(path):
        """让波浪号路径与模拟的用户目录保持一致。"""
        if path.parts and path.parts[0] == "~":
            return tmp_path.joinpath(*path.parts[1:])
        return original_expanduser(path)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(Path, "expanduser", expanduser)
    monkeypatch.setattr("minicrab.config.loader._current_config_path", None)


def test_runtime_dirs_follow_config_path(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance-a" / "config.json"
    monkeypatch.setattr("minicrab.config.paths.get_config_path", lambda: config_file)

    assert get_data_dir() == config_file.parent
    assert get_runtime_subdir("cron") == config_file.parent / "cron"
    assert get_cron_dir() == config_file.parent / "cron"
    assert get_logs_dir() == config_file.parent / "logs"


def test_media_dir_supports_channel_namespace(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance-b" / "config.json"
    monkeypatch.setattr("minicrab.config.paths.get_config_path", lambda: config_file)

    assert get_media_dir() == config_file.parent / "media"
    assert get_media_dir("telegram") == config_file.parent / "media" / "telegram"


def test_shared_and_legacy_paths_remain_global() -> None:
    assert get_cli_history_path() == Path.home() / ".minicrab" / "history" / "cli_history"
    assert get_bridge_install_dir() == Path.home() / ".minicrab" / "bridge"
    assert get_legacy_sessions_dir() == Path.home() / ".minicrab" / "sessions"


def test_workspace_path_is_explicitly_resolved() -> None:
    assert get_workspace_path() == Path.home() / ".minicrab" / "workspace"
    assert get_workspace_path("~/custom-workspace") == Path.home() / "custom-workspace"


def test_is_default_workspace_distinguishes_default_and_custom_paths() -> None:
    assert is_default_workspace(None) is True
    assert is_default_workspace(Path.home() / ".minicrab" / "workspace") is True
    assert is_default_workspace("~/custom-workspace") is False


def test_fresh_install_uses_minicrab_data():
    """新安装的配置、工作区和共享目录使用新名称。"""
    assert get_config_path() == Path.home() / ".minicrab" / "config.json"
    assert Config().agents.defaults.workspace == "~/.minicrab/workspace"
