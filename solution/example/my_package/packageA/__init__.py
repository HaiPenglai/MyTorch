from .file1 import greet
# TODO从packageB中导入一个hello函数，使得可以直接用packageA.hello()来调用
#《
from .packageB import hello
#》
__all__ = ['greet', 'hello']