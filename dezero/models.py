import dezero.functions as F
import dezero.layers as L
from dezero.layers import Model
import numpy as np


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

