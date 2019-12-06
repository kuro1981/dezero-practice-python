from dezero import Parameter, Function
import numpy as np
import dezero.functions as F

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

class Model(Layer):
    pass

class Linear(Layer):
    def __init__(self, in_size, out_size, nobias=False):
        super().__init__()

        I, O = in_size, out_size
        W_data = np.random.randn(I, O).astype(np.float32) * np.sqrt(1/I)
        self.W = Parameter(W_data, name='W')
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(O, dtype=np.float32), name='b')

    def __call__(self, x):
        y = F.linear(x, self.W, self.b)
        return y


