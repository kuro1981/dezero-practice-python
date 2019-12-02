import sys
sys.path.append("./")

import numpy as np
from dezero import Variable

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
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = goldstein(x, y)
        z.backward()

        self.assertEqual(x.grad, -5376.)
        self.assertEqual(y.grad, 8064.0)


if __name__ == '__main__':
    unittest.main()

