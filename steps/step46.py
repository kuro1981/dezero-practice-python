import sys
sys.path.append("./")

import numpy as np
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
import matplotlib.pyplot as plt
from dezero.models import MLP

# トイ・データセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

lr = 0.2
iters = 1000
hidden_size = 10

model = MLP((1, hidden_size, 1))
# optimizer = optimizers.SGD(lr)
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    print(loss)

