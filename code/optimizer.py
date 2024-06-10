import numpy as np


class AdamOptimizer:
    def __init__(self, network, epoch: int, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = network.layers

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = epoch
        self.epsilon = epsilon

        self.m_w, self.m_b = np.zeros(len(self.layers)), np.zeros(len(self.layers))
        self.v_w, self.v_b = np.zeros(len(self.layers)), np.zeros(len(self.layers))

    def step(self):
        for i, l in enumerate(self.layers):
            grad_w, m, v = self.calc_grad(l.weight_grad, self.m_w, self.v_w)
            self.m_w[i], self.v_w[i] = m, v

            grad_b, m, v = self.calc_grad(l.bias_grad, self.m_b, self.v_b)
            self.m_w[i], self.v_w[i] = m, v

            l.update(grad_w, grad_b)

    def calc_grad(self, grad, old_m, old_v):
        b1, b2 = self.beta1, self.beta2
        t, e = self.t, self.epsilon
        lr = self.learning_rate

        m = b1 * old_m + (1 - b1) * grad
        v = b2 * old_v + (1 - b2) * grad ** 2

        m1 = m / (1 - b1 ** (t + 1))
        v1 = m / (1 - b2 ** (t + 1))

        grad = lr * m1 / (np.sqrt(v1) + e)
        return grad, m, v
