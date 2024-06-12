import numpy as np


class AdamOptimizer:
    def __init__(self, network, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = network.layers

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.epsilon = epsilon

        self.m_w = [np.zeros(l.weights_shape) for l in self.layers]
        self.m_b = [np.zeros(l.bias_shape) for l in self.layers]
        self.v_w = [np.zeros(l.weights_shape) for l in self.layers]
        self.v_b = [np.zeros(l.bias_shape) for l in self.layers]

    def step(self):
        for i, l in enumerate(self.layers):
            grad_w, self.m_w[i], self.v_w[i] = self.calc_grad(l.weights_grad, self.m_w[i], self.v_w[i])
            grad_b, self.m_b[i], self.v_b[i] = self.calc_grad(l.bias_grad, self.m_b[i], self.v_b[i])
            l.update(grad_w, grad_b)
        self.t += 1

    def calc_grad(self, grad, old_m, old_v):
        b1, b2 = self.beta1, self.beta2
        t, e = self.t, self.epsilon
        lr = self.learning_rate

        m = b1 * old_m + (1 - b1) * grad
        v = b2 * old_v + (1 - b2) * grad ** 2

        m1 = m / (1 - b1 ** (t + 1))
        v1 = v / (1 - b2 ** (t + 1))

        grad = lr * m1 / (np.sqrt(v1) + e)
        return grad, m, v


class SimpleOptimizer:
    def __init__(self, network, learning_rate=0.001):
        self.layers = network.layers
        self.learning_rate = learning_rate

    def step(self):
        for i, l in enumerate(self.layers):
            grad_w = l.weights_grad
            grad_b = l.bias_grad
            lr = self.learning_rate
            l.update(grad_w * lr, grad_b * lr)
