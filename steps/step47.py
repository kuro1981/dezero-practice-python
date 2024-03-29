import sys
sys.path.append("./")

import numpy as np
from dezero import Variable, as_variable
import dezero.functions as F
from dezero.models import MLP

np.random.seed(0)

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


model = MLP((2, 10, 3))

x = Variable(np.array([0.2, -0.4]))
y = model(x)
p = softmax1d(y)
print(y)
print(p)

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])

y = model(x)
p = F.softmax(y)
print(y)
print(p)

loss = F.softmax_cross_entropy(y, t)
loss.backward()
print(loss)

