import numpy as np
from optimizer import *
from losses import *
from tqdm import tqdm


class NeuralNetwork:
    def __init__(self, layers, loss='mse'):
        self.optimizer = None
        self.layers = layers

        if loss == 'binary_crossentropy':
            self.loss = binary_cross_entropy
            self.loss_d = binary_cross_entropy_d
        else:
            self.loss = mse
            self.loss_d = mse_d

    def train(self, train_ds=(), valid_ds=(None, None), learning_rate=0.01, epochs=10, batch_size=1, optimizer='adam',
              verbose=True):
        if optimizer == 'grad_descent':
            self.optimizer = SimpleOptimizer(self, learning_rate)
        else:
            self.optimizer = AdamOptimizer(self, learning_rate)

        X, y = train_ds
        X_valid, y_valid = valid_ds

        for epoch in range(epochs):
            error = 0
            for i in tqdm(range(0, X.shape[0], batch_size)):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                y_pred = self._forward(X_batch)
                error += self.loss(y_batch, y_pred)

                grad = self.loss_d(y_batch, y_pred)
                self._backward(grad)

                self.optimizer.step()

            error /= (X.shape[0] // batch_size)

            if verbose:
                print("Epoch:", epoch, "Error:", error)

                if valid_ds is not (None, None):
                    acc, _ = self.validate(X_valid, y_valid)
                    print("Validation accuracy:", acc)

    def validate(self, X, y):
        output = self._forward(X)
        return np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1)), output

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
