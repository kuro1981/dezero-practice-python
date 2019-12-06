import numpy as np

class Optimizer:
    def setup(self, link):
        self.target = link
        return self

    def update(self):
        for param in self.target.params():
            self.update_one(param)

    def update_one(self, param):
        NotImplementedError()

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data

class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)
        
        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v
