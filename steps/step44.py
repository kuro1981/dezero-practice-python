import sys
sys.path.append("./")

import numpy as np
from dezero import Variable, Parameter
import dezero.functions as F
import matplotlib.pyplot as plt
import dezero.layers as L

# トイ・データセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

I, H, O = 1, 10, 1
l1 = L.Linear(I, H)
l2 = L.Linear(H, O)


def sigmoid(x):
    y = 1 / (1 + F.exp(-x))
    return y

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    # print(loss)

# グラフの描画
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')

t = Variable(np.arange(0, 1, .01)[:, np.newaxis])
y_pred = predict(t)
plt.plot(t.data, y_pred.data, color='r')
plt.show()

