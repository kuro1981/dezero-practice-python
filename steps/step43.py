import sys
sys.path.append("./")

import numpy as np
from dezero import Variable, Function, as_variable
import dezero.functions as F
import matplotlib.pyplot as plt

# トイ・データセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

I, H, O = 1, 10, 1
W1, b1 = Variable(0.01 * np.random.randn(I, H)), Variable(np.zeros(H))
W2, b2 = Variable(0.01 * np.random.randn(H, O)), Variable(np.zeros(O))

def sigmoid(x):
    y = 1 / (1 + F.exp(-x))
    return y

def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    # print(loss)

# グラフの描画
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')

t = Variable(np.arange(0, 1, .01)[:, np.newaxis])
y_pred = predict(t)
plt.plot(t.data, y_pred.data, color='r')
plt.show()

