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

    def update(self, w_grad, b_grad):
        pass

    @property
    def weight_grad(self):
        return 0

    @property
    def bias_grad(self):
        return 0

    def activate(self, X):
        if self.activation == 'relu':
            return np.maximum(X, 0)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        elif self.activation == 'tanh':
            return np.tanh(X)
        else:
            return X

    def activate_derivative(self, X):
        '''
        Derivative of the activation function
        X: input to the activation function
        '''
        if self.activation == 'relu':
            return np.where(X > 0, 1, 0)
        elif self.activation == 'sigmoid':
            sX = 1 / 1 + np.exp(-X)
            return sX * (1 - sX)
        elif self.activation == 'tanh':
            tanhX = np.tanh(X)
            return 1 - tanhX ** 2
        else:
            return X


class ConvLayer(Layer):
    def __init__(self, num_kernels: int, kernel_size: int, input_shape: tuple, activation='relu'):
        """
        CNN - layer
        Based on: https://www.youtube.com/watch?v=Lakz2MoHy6o
        """

        num_batches, height, width, channels = input_shape
        self.batch_num = num_batches
        self.num_kernels = num_kernels  # K
        self.input_shape = input_shape  # (B, H, W, C)
        self.num_channels = channels  # C
        self.output_shape = (
        num_batches, num_kernels, height - kernel_size + 1, width - kernel_size + 1)  # (B, K, H-KS+1, W+KS+1)

        self.kernels_shape = (num_kernels, channels, kernel_size, kernel_size)  # (K, C, KS, KS)
        self.kernels = np.random.randn(*self.kernels_shape)  # (K, C, KS, KS)
        self.biases = np.random.randn(*self.output_shape[1:])  # (K, H, W)

        self.kernels_grad = np.zeros(self.kernels_shape)
        self.biases_grad = np.zeros(self.output_shape[1:])

        self.activation = activation

    def forward(self, X):
        self.input = X  # (H, W, C)
        self.output = np.ndarray(self.output_shape)
        for b in range(self.batch_num):
            self.output[b] = np.copy(self.biases)
            for i in range(self.num_kernels):
                for j in range(self.num_channels):
                    # print(self.input[:, :, j].shape, self.kernels[i].shape)
                    self.output[b, i] += signal.correlate2d(self.input[b, :, :, j], self.kernels[i, j], mode='valid')
        self.derivative = self.activate_derivative(self.output)
        return self.activate(self.output)

    def backward(self, output_grad):
        output_grad = output_grad * self.derivative
        # output_grad.shape: (K, H-KS+1, W+KS+1)
        kernels_grad = np.zeros(self.kernels_shape)  # (K, C, KS, KS)
        input_grad = np.zeros(self.input_shape)  # (H, W, C)

        for i in range(self.num_kernels):
            for j in range(self.num_channels):
                kernels_grad[i, j] += signal.correlate2d(self.input[:, :, j], output_grad[i], 'valid')
                input_grad[:, :, j] += signal.convolve2d(output_grad[i], self.kernels[i, j], 'full')

        self.kernels_grad = kernels_grad
        self.biases_grad = output_grad
        # self.kernels -= self.learning_rate * kernels_grad
        # self.biases -= self.learning_rate * output_grad
        return input_grad

    def update(self, w_grad, b_grad):
        self.kernels -= w_grad
        self.biases -= b_grad

    @property
    def weights(self):
        return self.kernels_grad

    @property
    def bias(self):
        return self.bias_grad


class PoolLayer(Layer):
    def __init__(self, num_filters: int, filter_size: tuple, stride, padding, input_shape):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape


class MaxPoolLayer(Layer):
    def __init__(self, pool_size, input_shape, stride=None, pool_type='max'):
        self.pool_size = pool_size
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride
        self.input_shape = input_shape
        self.pool_type = pool_type
        self.output_shape = (input_shape[0], (input_shape[1] - self.pool_size[0]) // self.stride[0] + 1,
                             (input_shape[2] - self.pool_size[1]) // self.stride[1] + 1, input_shape[3])

    def _compress_indices(self):
        return np.mean(self.indices, axis=0)

    def forward(self, X):
        # implement max pooling and save the indices of the max values
        print(X.shape)
        self.output = np.zeros(self.output_shape)

        (batch, x, y, channel) = X.shape

        self.indices = np.zeros(self.input_shape)

        for i in range(0, x - self.pool_size[0] + 1, self.stride[0]):
            for j in range(0, y - self.pool_size[1] + 1, self.stride[1]):
                for c in range(channel):
                    self.output[:, i // self.stride[0], j // self.stride[1], c] = np.max(
                        X[:, i:i + self.pool_size[0], j:j + self.pool_size[1], c], axis=(1, 2))
                    self.indices[:, i:i + self.pool_size[0], j:j + self.pool_size[1], c] = np.where(
                        X[:, i:i + self.pool_size[0], j:j + self.pool_size[1], c] == self.output[:, i // self.stride[0],
                                                                                     j // self.stride[1], c].reshape(-1,
                                                                                                                     1,
                                                                                                                     1),
                        1, 0)

        self.indices = self._compress_indices()
        print(self.indices.shape)

        return self.output

    def backward(self, delta):
        # implement backpropagation for max pooling using the saved indices
        delta = np.repeat(delta, self.pool_size[0], axis=0)

        delta = np.repeat(delta, self.pool_size[1], axis=1)

        delta_out = delta * self.indices

        return delta_out


class MeanPoolLayer(Layer):
    def __init__(self, pool_size, input_shape, stride=None, pool_type='max'):
        self.pool_size = pool_size
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride
        self.input_shape = input_shape
        self.pool_type = pool_type

        self.output_shape = (input_shape[0], (input_shape[1] - self.pool_size[0]) // self.stride[0] + 1,
                             (input_shape[2] - self.pool_size[1]) // self.stride[1] + 1, input_shape[3])

    def forward(self, X):
        # implement mean pooling
        self.output = np.zeros(self.output_shape)

        reshaped_input = X.reshape(X.shape[0], X.shape[1] // self.pool_size[0], self.pool_size[0],
                                   X.shape[2] // self.pool_size[1], self.pool_size[1], X.shape[3])
        output = reshaped_input.mean(axis=(2, 4))
        reshaped_input = reshaped_input.transpose(0, 1, 3, 5, 2, 4)
        # self.indices = each neuron equal weight
        return self.activate(output)

    def backward(self, output_grad):
        # implement backpropagation for mean pooling

        output_grad = np.repeat(output_grad, self.pool_size[0], axis=0)
        output_grad = np.repeat(output_grad, self.pool_size[1], axis=1)

        output_grad = output_grad / (self.pool_size[0] * self.pool_size[1])

        return output_grad


class FlattenLayer(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], np.prod(input_shape[1:]))

    def forward(self, X):
        self.input = X
        self.output = X.reshape(X.shape[0], -1)

        return self.output

    def backward(self, output_grad):
        return output_grad.reshape(self.input_shape)


class DenseLayer(Layer):
    def __init__(self, units, input_shape, activation='relu'):
        self.units = units
        self.input_shape = input_shape
        self.activation = activation

        self.weights = np.random.randn(input_shape[1], units) - 0.5
        self.biases = np.random.randn(units) - 0.5

    def forward(self, X):
        self.input = X

        self.output = np.dot(X, self.weights) + self.biases
        self.derivative = self.activate_derivative(self.output)
        return self.activate(self.output)

    def backward(self, output_grad):
        output_grad = output_grad * self.derivative
        input_error = np.dot(output_grad, self.weights.T)
        weights_error = np.dot(self.input.T, output_grad)

        self.weights_grad = weights_error
        self.biases_grad = output_grad

        return input_error

    @property
    def weights_grad(self):
        return self.weights_grad

    @property
    def bias_grad(self):
        return self.biases_grad

    def update(self, w_grad, b_grad):
        self.weights -= w_grad
        self.biases -= b_grad
