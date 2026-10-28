"""兼容旧的 python -m nanobot，实际运行 minicrab 命令行。"""

from minicrab.cli.commands import app

if __name__ == "__main__":
    app()
