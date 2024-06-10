import numpy as np
from scipy import signal


class Layer:
    def __init__(self, input_shape):
        self.activation = None
        self.delta = None
        self.id = None

        self.input_shape = input_shape

    def set_id(self, id):
        self.id = id

    def forward(self, X):
        pass

    def backward(self, delta):
        pass

    def set_id(self, id):
        self.id = id

    def get_output(self):
        return self.output


# https://www.youtube.com/watch?v=Lakz2MoHy6o
class ConvLayer(Layer):
    def __init__(self, num_kernels: int, kernel_size: int, input_shape: tuple, learning_rate=0.001):
        """
        CNN - layer
        Based on: https://www.youtube.com/watch?v=Lakz2MoHy6o
        """

        height, width, channels = input_shape
        self.num_kernels = num_kernels  # K
        self.input_shape = input_shape  # (H, W, C)
        self.num_channels = channels  # C
        self.output_shape = (num_kernels, height - kernel_size + 1, width - kernel_size + 1)  # (K, H-KS+1, W+KS+1)
        self.kernels_shape = (num_kernels, channels, kernel_size, kernel_size)  # (K, C, KS, KS)
        self.kernels = np.random.randn(*self.kernels_shape)  # (K, C, KS, KS)
        self.biases = np.random.randn(*self.output_shape)  # (K, H, W)

        self.learning_rate = learning_rate

    def forward(self, X):
        self.input = X  # (H, W, C)
        self.output = np.copy(self.biases)
        for i in range(self.num_kernels):
            for j in range(self.num_channels):
                # print(self.input[:, :, j].shape, self.kernels[i].shape)
                self.output[i] += signal.correlate2d(self.input[:, :, j], self.kernels[i, j], mode='valid')
        return self.output

    def backward(self, output_grad):
        # output_grad.shape: (K, H-KS+1, W+KS+1)
        kernels_grad = np.zeros(self.kernels_shape)  # (K, C, KS, KS)
        input_grad = np.zeros(self.input_shape)  # (H, W, C)

        for i in range(self.num_kernels):
            for j in range(self.num_channels):
                kernels_grad[i, j] += signal.correlate2d(self.input[:, :, j], output_grad[i], 'valid')
                input_grad[:, :, j] += signal.convolve2d(output_grad[i], self.kernels[i, j], 'full')

        self.kernels -= self.learning_rate * kernels_grad
        self.biases -= self.learning_rate * output_grad
        return input_grad


class PoolLayer(Layer):
    def __init__(self, pool_size, stride, input_shape, pool_type='max'):
        self.pool_size = pool_size
        self.stride = stride
        self.input_shape = input_shape
        self.pool_type = pool_type

    def forward(self, X):
        pass

    def backward(self, delta):
        pass


class FlattenLayer(Layer):
    def forward(self, X):
        pass

    def backward(self, delta):
        pass


class DenseLayer(Layer):
    def __init__(self, units, input_shape, activation='relu'):
        self.units = units
        self.input_shape = input_shape
        self.activation = activation

        self.weights = np.random.randn(input_shape[1], units) - 0.5
        self.biases = np.random.randn(units) - 0.5

    def activate(self, X):
        if self.activation == 'relu':
            return np.maximum(X, 0)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        elif self.activation == 'tanh':
            return np.tanh(X)
        else:
            return X

    def forward(self, X):
        self.input = X

        self.output = np.dot(X, self.weights) + self.biases

        return self.activate(self.output)

    def backward(self, delta):
        input_error = np.dot(delta, self.weights.T)
        weights_error = np.dot(self.input.T, delta)

        self.weights -= self.learning_rate * weights_error
        self.biases -= self.learning_rate * delta

        return input_error
