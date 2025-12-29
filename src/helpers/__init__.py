# helpers/__init__.py
from os.path import dirname, basename, isfile, join
import glob

# 移除这两行，避免循环导入
# from .DiffKGReader import DiffKGReader
# from .DiffKGRunner import DiffKGRunner

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')
]

# 创建一个延迟导入的字典
_import_cache = {}

def __getattr__(name):
    """延迟导入模块"""
    if name not in _import_cache:
        if name == 'DiffKGReader':
            from .DiffKGReader import DiffKGReader
            _import_cache[name] = DiffKGReader
        elif name == 'DiffKGRunner':
            from .DiffKGRunner import DiffKGRunner
            _import_cache[name] = DiffKGRunner
        elif name == 'BaseReader':
            from .BaseReader import BaseReader
            _import_cache[name] = BaseReader
        elif name == 'BaseRunner':
            from .BaseRunner import BaseRunner
            _import_cache[name] = BaseRunner
        else:
            raise AttributeError(f"module 'helpers' has no attribute '{name}'")
    return _import_cache[name]

# 可选：导出模块名
__all__ = ['BaseReader', 'BaseRunner', 'DiffKGReader', 'DiffKGRunner']