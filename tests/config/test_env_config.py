"""验证命令行使用的 minicrab 环境变量配置入口。"""

from minicrab.config.schema import Config


def test_environment_model_setting(monkeypatch):
    """启动时可通过环境变量指定模型。"""
    monkeypatch.setenv("MINICRAB_AGENTS__DEFAULTS__MODEL", "configured-model")
    assert Config().agents.defaults.model == "configured-model"
