from __future__ import print_function
import sys

try:
    from .cy_impl import memory_efficient_inner1d
except ImportError:
    print(
        'Cannot import cy_impl.memory_efficient_inner1d. py_impl is used',
        file=sys.stderr
    )
    from .py_impl import memory_efficient_inner1d
