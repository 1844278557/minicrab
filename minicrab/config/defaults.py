"""提供 minicrab 的默认配置目录和工作区，供配置与运行路径共用。"""

from pathlib import Path

APP_DATA_DIR = ".minicrab"


def get_default_data_dir() -> Path:
    """返回 minicrab 用户数据目录。"""
    return Path.home() / APP_DATA_DIR


def get_default_workspace() -> str:
    """返回与所选默认数据目录一致的工作区配置值。"""
    return f"~/{get_default_data_dir().name}/workspace"
