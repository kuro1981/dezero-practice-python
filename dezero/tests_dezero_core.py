import sys
sys.path.append("./")

import numpy as np
import unittest
from dezero import Variable, using_config, no_grad
from dezero.core import add, mul, div, pow
import dezero.functions as F
from dezero.utils import numerical_diff
from contextlib import contextmanager

class DezeroCoreTest(unittest.TestCase):

    def setUp(self):
        # 念の為
        using_config('enable_backprop', True)

    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type:
            raise self.failureException('{} raised'.format(exc_type.__name__))
    
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = x ** 2
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = x ** 2
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad.data, expected)

    def test_add_check(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        # それぞれ変数として与える。
        y = add(x0, x1)
        self.assertEqual(y.data, np.array(5))

    def test_add_same_variable(self):
        # 1層
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()
        self.assertEqual(x.grad.data, 2.)
        
        # 2層
        x = Variable(np.array(3.0))
        y = add(add(x, x), x)
        y.backward()
        self.assertEqual(x.grad.data, 3.)

        # 再利用層
        x.cleargrad()
        y = add(add(x, x), x)
        y.backward()
        self.assertEqual(x.grad.data, 3.)

        # 分岐有
        x = Variable(np.array(2.0))
        a = x ** 2
        y = add(a ** 2, a ** 2)
        y.backward()
        self.assertEqual(y.data, 32.)
        self.assertEqual(x.grad.data, 64.)

    def test_priority_check(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = add(x ** 2, y ** 2)
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
        self.assertIsNotNone(x0.grad.data)
        self.assertIsNotNone(x1.grad.data)
       
        # retain_grad True
        x0.cleargrad()
        x1.cleargrad()
        y.backward(retain_grad = True)
        self.assertIsNotNone(y.grad.data)
        self.assertIsNotNone(t.grad.data)
        
    def test_config_enable_backprop_check(self):
        # backprop True
        x = Variable(np.ones((100,100,100)))
        y = x ** 3
        with self.assertNotRaises(Exception):
            y.backward()    

        # backprop False
        with no_grad():
            x.cleargrad()
            y = x ** 3
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
        self.assertEqual(a.grad.data, 2.)
        self.assertEqual(b.grad.data, 3.)

        y.cleargrad()
        a.cleargrad()
        y = a * np.array(2.0) + 1.0
        y.backward()
        self.assertEqual(y.data, np.array(7.))
        self.assertEqual(a.grad.data, 2.)
        self.assertEqual(b.grad.data, 3.)
        
    def test_underscore_add_mul(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(2.0))
        
        y = a * b
        y.backward()
        
        self.assertEqual(y.data, np.array(6.))
        self.assertEqual(a.grad.data, 2.)
        self.assertEqual(b.grad.data, 3.)

        y.cleargrad()
        y = (a * b) + c
    
        self.assertEqual(y.data, np.array(8.))
   
    def test_neg(self):
        x = Variable(np.array(3.0))
        y = -x
        y.backward()
        self.assertEqual(y.data, np.array(-3.0))
        self.assertEqual(x.grad.data, -1.)

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
        self.assertEqual(x.grad.data, 5.0 * 3 ** (4))
    
    def test_second_diff(self):
        def f(x):
            y = x ** 4 - 2 * x ** 2
            return y
        x = Variable(np.array(2.0))
        y = f(x)
        y.backward(create_graph=True)
        self.assertEqual(x.grad.data, 24.)

        gx = x.grad
        x.cleargrad()
        gx.backward()
        self.assertEqual(x.grad.data, 44.)
    
    def test_tanh(self):
        x = Variable(np.array(2.0))
        y = F.tanh(x) 
        y.backward()

        expected_y = np.tanh(np.array(2.0))
        self.assertEqual(y.data, expected_y)
        self.assertEqual(x.grad.data, 1 - expected_y ** 2)

    def test_reshape(self):
        x = Variable(np.array([[1.,2.,3.],[4.,5.,6.]]))
        y = F.reshape(x,(6,))
        y.backward(retain_grad=True)
        
        self.assertEqual(y.shape, (6,))
        self.assertEqual(x.grad.shape, (2,3))

        x = Variable(np.random.randn(1, 2, 3))
        with self.assertNotRaises(Exception):
            y = x.reshape((2,3))

    def test_reshape(self):
        x = Variable(np.array([[1.,2.,3.],[4.,5.,6.]]))
        y = F.transpose(x)
        y.backward(retain_grad=True)
        
        self.assertEqual(y.shape, (3,2))
        self.assertEqual(x.grad.shape, (2,3))


if __name__ == '__main__':
    unittest.main()
