import numpy as np
import weakref
import contextlib

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

class Variable:
    __array_priority__ = 200
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.priority = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype
    
    def set_creator(self, func):
        self.creator = func
        self.priority = func.priority + 1 

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.priority)
        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable({})'.format(p)

#    def __mul__(self, other):
#        return mul(self, other)
#
#    def __add__(self, other):
#        return add(self, other)
    
class Function:
    def __call__(self, *inputs) -> list:
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs= [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
        #if True:
            self.priority = max([x.priority for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

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

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy / (-x0 / x1**2)
        return gx0, -gx1

class Pow(Function):
    def __init__(self, c):
          self.c = c

    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c 
        gx = c * x ** (c-1) * gy
        return gx


def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    # add radd で吸収
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

def pow(x, c):
    return Pow(c)(x)

Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__add__ = add
Variable.__radd__ = add
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow


# utility

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

# TEST

import unittest
from contextlib import contextmanager

class SquareTest(unittest.TestCase):

    def setUp(self):
        # 念の為
        Config.enable_backprop = True

    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type:
            raise self.failureException('{} raised'.format(exc_type.__name__))
    
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

    def test_add_check(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        # それぞれ変数として与える。
        y = add(x0, x1)
        self.assertEqual(y.data, 5.)

    def test_add_square_backward(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = add(square(x), square(y))
        z.backward()
        self.assertEqual(z.data, 13.)
        self.assertEqual(x.grad, 4.)
        self.assertEqual(y.grad, 6.)

    def test_add_same_variable(self):
        # 1層
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()
        self.assertEqual(x.grad, 2.)
        
        # 2層
        x = Variable(np.array(3.0))
        y = add(add(x, x), x)
        y.backward()
        self.assertEqual(x.grad, 3.)

        # 再利用層
        x.cleargrad()
        y = add(add(x, x), x)
        y.backward()
        self.assertEqual(x.grad, 3.)

        # 分岐有
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        self.assertEqual(y.data, 32.)
        self.assertEqual(x.grad, 64.)

    def test_priority_check(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = add(square(x), square(y))
        z.backward()
        self.assertEqual(x.priority, 0)
        self.assertEqual(z.priority, 2)
        self.assertEqual(z.creator.priority, 1)

    def test_variable_retain_grad_check(self):
        x0 = Variable(np.array(1.))
        x1 = Variable(np.array(1.))
        # retain_grad False
        t = add(x0, x1)
        y = add(x0, t)
        y.backward()
        
        
        self.assertIsNone(y.grad)
        self.assertIsNone(t.grad)
        self.assertIsNotNone(x0.grad)
        self.assertIsNotNone(x1.grad)
       
        # retain_grad True
        x0.cleargrad()
        x1.cleargrad()
        y.backward(retain_grad = True)
        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(t.grad)
        
    def test_config_enable_backprop_check(self):
        # backprop True
        x = Variable(np.ones((100,100,100)))
        y = square(square(square(x)))
        with self.assertNotRaises(Exception):
            y.backward()    

        # backprop False
        with no_grad():
            x.cleargrad()
            y = square(square(square(x)))
            with self.assertRaises(Exception):
                y.backward()    
    
    def test_variable_shape(self):
        x = Variable(np.array([[1,2,3],[4,5,6]]))
        self.assertEqual(x.shape, x.data.shape)
    
    def test_variable_size(self):
        x = Variable(np.array([[1,2,3],[4,5,6]]))
        self.assertEqual(x.size, x.data.size) 
    
    def test_variable_dtype(self):
        x = Variable(np.array([[1,2,3],[4,5,6]]))
        self.assertEqual(x.dtype, x.data.dtype)
    
    def test_variable_length(self):
        x = Variable(np.array([[1,2,3],[4,5,6]]))
        self.assertEqual(len(x), len(x.data))

    def test_mul(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(2.0))

        y = add(mul(a, b), c)
        y.backward()
        self.assertEqual(y.data, np.array(8.))
        self.assertEqual(a.grad, 2.)
        self.assertEqual(b.grad, 3.)

        y.cleargrad()
        a.cleargrad()
        y = a * np.array(2.0) + 1.0
        y.backward()
        self.assertEqual(y.data, np.array(7.))
        self.assertEqual(a.grad, 2.)
        self.assertEqual(b.grad, 3.)
        
    def test_underscore_add_mul(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(2.0))
        
        y = a * b
        y.backward()
        
        self.assertEqual(y.data, np.array(6.))
        self.assertEqual(a.grad, 2.)
        self.assertEqual(b.grad, 3.)

        y.cleargrad()
        y = (a * b) + c
    
        self.assertEqual(y.data, np.array(8.))
   
    def test_neg(self):
        x = Variable(np.array(3.0))
        y = -x
        y.backward()
        self.assertEqual(y.data, np.array(-3.0))
        self.assertEqual(x.grad, -1.)

    def test_sub(self):
        x = Variable(np.array(3.0))
        y1 = 2.0 - x
        y2 = x - 1.0
        self.assertEqual(y1.data, np.array(-1.0))
        self.assertEqual(y2.data, np.array(2.0))
    
    def test_div(self):
        x = Variable(np.array(3.0))
        y1 = 3.0 / x
        y2 = x / 1.0
        self.assertEqual(y1.data, np.array(1.0))
        self.assertEqual(y2.data, np.array(3.0))

    def test_pow(self):
        x = Variable(np.array(3.0))
        y = x ** 5
        y.backward()
        self.assertEqual(y.data, np.array(3.0 ** 5))
        self.assertEqual(x.grad, 5.0 * 3 ** (4))
        
        
if __name__ == '__main__':
    unittest.main()
