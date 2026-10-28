"""旧包名的兼容入口，将旧插件的导入转向 minicrab，实际代码均在新包中。"""

import importlib
import importlib.abc
import importlib.util
import sys

from minicrab import __app_name__, __logo__, __version__

__all__ = ["__app_name__", "__logo__", "__version__"]


class _AliasLoader(importlib.abc.Loader):
    """让旧导入和新导入指向同一个模块，避免插件类型不一致。"""

    def __init__(self, target_name):
        """保存要导入的新模块名称。"""
        self.target_name = target_name
        self.target_spec = None

    def create_module(self, spec):
        """复用新模块，不执行两份模块代码。"""
        module = importlib.import_module(self.target_name)
        self.target_spec = module.__spec__
        return module

    def exec_module(self, module):
        """保留新模块的资源加载信息，不让兼容入口覆盖它。"""
        module.__spec__ = self.target_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    """只处理 nanobot 子模块，其余导入交给 Python。"""

    def find_spec(self, fullname, path=None, target=None):
        """查找对应的新模块，并为旧模块名创建别名。"""
        if not fullname.startswith("nanobot.") or fullname == "nanobot.__main__":
            return None
        target_name = "minicrab" + fullname[len("nanobot"):]
        spec = importlib.util.find_spec(target_name)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(target_name),
            is_package=spec.submodule_search_locations is not None,
        )


sys.meta_path.insert(0, _AliasFinder())
