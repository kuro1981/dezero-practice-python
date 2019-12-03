import sys
sys.path.append("./")

import numpy as np
from dezero import Variable, Function
from dezero.utils import get_dot_graph
import math

def sphere(x, y):
    y = x ** 2 + y ** 2
    return y

def matyas(x, y):
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z

def goldstein(x, y):
    y = (1 +(x+y+1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y -36*x*y + 27*y**2))
    return y

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(10000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


import unittest

class OptimizationTest(unittest.TestCase):
    def test_sphere(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(2.0))
        z = sphere(x, y)
        z.backward()

        self.assertEqual(x.grad, 2.)
        self.assertEqual(y.grad, 4.)

    def test_matyas(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = matyas(x, y)
        z.backward()
        
    def test_goldstein(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        y= goldstein(x0, x1)
        y.backward()

        self.assertEqual(x0.grad, -5376.)
        self.assertEqual(x1.grad, 8064.0)
        
        x0.name = 'x0'
        x1.name = 'x1'
        y.name = 'y'
        # with open('graph.dot',"w") as f:
        #     f.write(get_dot_graph(y))

class AdditionalFuncTest(unittest.TestCase):
    def test_sin(self):
        _x = np.array(np.pi/4)
        x = Variable(_x)
        y = sin(x)
        y.backward()

        self.assertEqual(y.data, np.sin(_x))
        self.assertEqual(x.grad, np.cos(_x))
    
    def test_mysin(self):
        _x = np.array(np.pi/4)
        x0 = Variable(_x)
        y0 = sin(x0)
        y0.backward()

        x1 = Variable(_x)
        y1 = my_sin(x1)
        y1.backward()

        self.assertTrue(np.allclose(y0.data, y1.data))
        self.assertTrue(np.allclose(x0.grad, x1.grad))

if __name__ == '__main__':
    unittest.main()

