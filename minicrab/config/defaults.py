"""统一默认数据目录，供配置和运行路径使用，并兼容已有 nanobot 数据。"""

from pathlib import Path

APP_DATA_DIR = ".minicrab"
LEGACY_DATA_DIR = ".nanobot"


def get_default_data_dir() -> Path:
    """新环境使用 minicrab；已有旧配置且没有新配置时继续使用旧目录。"""
    current = Path.home() / APP_DATA_DIR
    legacy = Path.home() / LEGACY_DATA_DIR
    if not (current / "config.json").is_file() and (legacy / "config.json").is_file():
        return legacy
    return current


def get_default_workspace() -> str:
    """返回与所选默认数据目录一致的工作区配置值。"""
    return f"~/{get_default_data_dir().name}/workspace"
