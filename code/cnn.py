import numpy as np
from optimizer import *
from losses import *


class NuralNetwork:
    def __init__(self, loss='mse'):
        self.layers = []

        if loss == 'binary_crossentropy':
            self.loss = binary_cross_entropy
            self.loss_d = binary_cross_entropy_d
        else:
            self.loss = mse
            self.loss_d = mse_d

    def add(self, layer):
        self.layers.append(layer)

        # layer.set_id(self.next_layer)
        # self.next_layer += 1

    def train(self, X, y, learning_rate=0.01, epochs=10, batch_size=1, verbose=True):
        batch_learning_rate = learning_rate / batch_size

        self.error = []
        self.optimizer = SimpleOptimizer(self, learning_rate=learning_rate)  # Or batch_lr?

        for epoch in range(epochs):
            error = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                y_pred = self._forward(X_batch)
                error += self.loss(y_batch, y_pred)
                print("loss", error)

                grad = self.loss_d(y_batch, y_pred)
                self._backward(grad)

                self.optimizer.step()

            error /= batch_size
            self.error.append(error)

            if verbose:
                print("Epoch", epoch, "Error", self.error[-1])

    def predict(self, X):
        X = X.reshape(1, *X.shape)
        return self._forward(X)[0]

    def evaluate(self, X, y):
        output = self.predict(X)
        return np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1)), output

    def _forward(self, X):
        for l in self.layers:
            X = l.forward(X)
        return X

    def _backward(self, grad):
        for l in reversed(self.layers):
            grad = l.backward(grad)
        return grad
        # self.error.append(np.mean(np.abs(y)))