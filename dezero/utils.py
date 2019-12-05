import numpy as np
import weakref
import contextlib
from dezero.core_simple import Function, Variable

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

def _dot_var(v):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = ''
    if hasattr(v, 'name') and v.name is not None:
        name = v.name
    return dot_var.format(id(v), name)

def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y())) # y is weakref
    return txt

def get_dot_graph(y):
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(y.creator)
    txt = _dot_var(y)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'

# =============================================================================
# Utility functions (numpy magic)
# =============================================================================
def sum_to(x, shape):
    """x が shape の形状になるように和を求める。
    Parameters
    ----------
    x : numpy.ndarray
    shape : None or int or tuple of ints
    Returns
    -------
    y : numpy.ndarray
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """dezero.functions.sum関数の逆伝播で伝わる勾配を適切な形状に変換する。
    Parameters
    ----------
    gy : dezero.Variable
        逆伝播で出力側から伝わる勾配
    x_shape : tuple
        順伝播のsum関数で使用した入力変数の形状
    axis : None or int or tuple of ints
        順伝播のsum関数の引数で指定した axis
    keepdims : bool
        順伝播のsum関数の引数で指定した keepdims
    Returns
    -------
    gy : dezero.Variable
        形状変換後の勾配
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not hasattr(axis, 'len'):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy

