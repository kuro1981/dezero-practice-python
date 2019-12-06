import sys
sys.path.append("./")

import numpy as np
from dezero import Variable, Parameter, Layer
import dezero.functions as F
import matplotlib.pyplot as plt
import dezero.layers as L 
from dezero.layers import Model

# トイ・データセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

I, H, O = 1, 10, 1

class TwoLayerNet(Model):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(in_size, hidden_size)
        self.l2 = L.Linear(hidden_size, out_size)

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        y = self.l2(h)
        return y

#model = Model()
#model.l1 = L.Linear(I, H)
#model.l2 = L.Linear(H, O)

def predict(x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y

model = TwoLayerNet(1, H, 1)

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    # print(loss)

class MLP(Model):
    def __init__(self, sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layer = L.Linear(in_size, out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)

model = MLP((1, 10, 20, 30, 1))

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    print(loss)


