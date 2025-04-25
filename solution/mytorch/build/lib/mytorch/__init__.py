# 版本信息
__version__ = "0.1.3"

# 暴露主要子模块
from . import mynn
from .quantization import TensorQ

# 可选：快捷导入常用组件（根据需求选择）
__all__ = ['mynn']  # 推荐仅暴露子模块命名空间