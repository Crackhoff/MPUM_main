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
    def weights_grad(self):
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
            sX = 1 / (1 + np.exp(-X))
            return sX * (1 - sX)
        elif self.activation == 'tanh':
            tanhX = np.tanh(X)
            return 1 - tanhX ** 2
        else:
            return np.ones(X.shape)


class ConvLayer(Layer):
    def __init__(self, num_kernels: int, kernel_size: int, input_shape: tuple, activation='relu'):
        """
        CNN - layer
        Based on: https://www.youtube.com/watch?v=Lakz2MoHy6o
        """
        channels, height, width, = input_shape
        # self.batch_num = num_batches
        self.num_kernels = num_kernels  # K
        self.input_shape = input_shape  # (B, H, W, C)
        self.num_channels = channels  # C
        self.output_shape = (num_kernels, height - kernel_size + 1, width - kernel_size + 1)  # (B, K, H-KS+1, W+KS+1)

        self.kernels_shape = (num_kernels, channels, kernel_size, kernel_size)  # (K, C, KS, KS)
        self.kernels = np.random.randn(*self.kernels_shape)  # (K, C, KS, KS)
        self.biases = np.random.randn(*self.output_shape)  # (K, H, W)

        self.kernels_grad = np.zeros(self.kernels_shape)
        self.biases_grad = np.zeros(self.output_shape)

        self.activation = activation

    def forward(self, X):
        batches_num = X.shape[0]
        self.input = X  # (H, W, C)
        self.output = np.ndarray((batches_num, *self.output_shape))
        for b in range(batches_num):
            self.output[b] = np.copy(self.biases)
            # print(self.num_kernels)
            for i in range(self.num_kernels):
                for j in range(self.num_channels):
                    # print(self.input[:, :, j].shape, self.kernels[i].shape)
                    self.output[b, i] += signal.correlate2d(self.input[b, j], self.kernels[i, j], mode='valid')

        self.derivative = self.activate_derivative(self.output)
        return self.activate(self.output)

    def backward(self, output_grad):
        batches_num = self.input.shape[0]
        output_grad = output_grad * np.mean(self.derivative, axis=0)
        # print(output_grad.shape)
        # output_grad.shape: (K, H-KS+1, W+KS+1)
        kernels_grad = np.zeros((batches_num, *self.kernels_shape))  # (K, C, KS, KS)
        input_grad = np.zeros(self.input_shape)  # (H, W, C)

        for i in range(self.num_kernels):
            for j in range(self.num_channels):
                for b in range(batches_num):
                    # print("xd",self.input.shape, output_grad.shape)
                    kernels_grad[b, i, j] = signal.correlate2d(self.input[b, j], output_grad[i], 'valid')
                # print(signal.convolve2d(output_grad[i], self.kernels[i, j], 'full').shape, input_grad.shape)
                input_grad[j] += signal.convolve2d(output_grad[i], self.kernels[i, j], 'full')

        self.kernels_grad = np.mean(kernels_grad, axis=0)
        self.biases_grad = output_grad
        # self.kernels -= self.learning_rate * kernels_grad
        # self.biases -= self.learning_rate * output_grad
        return input_grad

    def update(self, w_grad, b_grad):
        self.kernels -= w_grad
        self.biases -= b_grad

    @property
    def weights_grad(self):
        return self.kernels_grad

    @property
    def bias_grad(self):
        return self.biases_grad


class PoolLayer(Layer):
    def __init__(self, num_filters: int, filter_size: tuple, stride, padding, input_shape):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape


class MaxPoolLayer(Layer):
    def __init__(self, input_shape, pool_size, stride=None):
        self.pool_size = pool_size
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], (input_shape[1] - self.pool_size[0]) // self.stride[0] + 1,
                             (input_shape[2] - self.pool_size[1]) // self.stride[1] + 1) # (C, H, W)

    def _compress_indices(self):
        return np.mean(self.indices, axis=0)

    def forward(self, X):
        # implement max pooling and save the indices of the max values
        # print(X.shape)
        (batch, channel, x, y) = X.shape
        self.output = np.zeros((batch, *self.output_shape))

        
        self.indices = np.zeros(X.shape)
        for b in range(batch):
            for c in range(channel):
                for i in range(0, x - self.pool_size[0] + 1, self.stride[0]):
                    for j in range(0, y - self.pool_size[1] + 1, self.stride[1]):
                        self.output[b, c, i // self.stride[0], j // self.stride[1]] = np.max(
                            X[b, c, i:i + self.pool_size[0], j:j + self.pool_size[1]])
                        self.indices[b, c, i:i + self.pool_size[0], j:j + self.pool_size[1]] = np.where(
                            X[b, c, i:i + self.pool_size[0], j:j + self.pool_size[1]] == self.output[b, c, i // self.stride[0],
                                                                                       j // self.stride[1],
                                                                                       ].reshape(-1, 1, 1), 1, 0)
        
        # for i in range(0, x - self.pool_size[0] + 1, self.stride[0]):
        #     for j in range(0, y - self.pool_size[1] + 1, self.stride[1]):
        #         for c in range(channel):
        #             self.output[:, i // self.stride[0], j // self.stride[1], c] = np.max(
        #                 X[:, i:i + self.pool_size[0], j:j + self.pool_size[1], c], axis=(1, 2))
        #             self.indices[:, i:i + self.pool_size[0], j:j + self.pool_size[1], c] = np.where(
        #                 X[:, i:i + self.pool_size[0], j:j + self.pool_size[1], c] == self.output[:, i // self.stride[0],
        #                                                                              j // self.stride[1], c].reshape(-1,
        #                                                                                                              1,
        #                                                                                                              1),
        #                 1, 0)

        self.indices = self._compress_indices()

        return self.output

    def backward(self, output_grad):
        # implement backpropagation for max pooling using the saved indices
        output_grad = np.repeat(output_grad, self.pool_size[0], axis=1)

        output_grad = np.repeat(output_grad, self.pool_size[1], axis=2)

        output_grad = output_grad * self.indices

        return output_grad


class MeanPoolLayer(Layer):
    def __init__(self, input_shape, pool_size, stride=None):
        self.pool_size = pool_size
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], (input_shape[1] - self.pool_size[0]) // self.stride[0] + 1,
                             (input_shape[2] - self.pool_size[1]) // self.stride[1] + 1) # (C, H, W)

    def forward(self, X):
        # implement mean pooling
        self.output = np.zeros(self.output_shape)
        
        batches_num = X.shape[0]

        reshaped_input = X.reshape(X.shape[0], X.shape[1], X.shape[2] // self.pool_size[0], self.pool_size[0],
                                   X.shape[3] // self.pool_size[1], self.pool_size[1])
        output = reshaped_input.mean(axis=(3, 5))
        
        # self.indices = each neuron equal weight
        return output

    def backward(self, output_grad):
        # implement backpropagation for mean pooling
        # 
        output_grad = np.repeat(output_grad, self.pool_size[0], axis=1)
        output_grad = np.repeat(output_grad, self.pool_size[1], axis=2)

        output_grad = output_grad / (self.pool_size[0] * self.pool_size[1])

        return output_grad


class FlattenLayer(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], np.prod(input_shape[1:]))

    def forward(self, X):
        self.input = X
        self.output = X.reshape(X.shape[0], -1, 1)
        return self.output

    def backward(self, output_grad):
        return output_grad.reshape(self.input_shape)


class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

        self._weights_grad = None
        self._biases_grad = None

    def forward(self, X):
        self.input = X
        batches_num = X.shape[0]

        self.output = np.ndarray((batches_num, self.output_size, 1))
        for i in range(batches_num):
            self.output[i] = np.dot(self.weights, self.input[i]) + self.biases
        self.derivative = self.activate_derivative(self.output)
        return self.activate(self.output)

    def backward(self, output_grad):
        output_grad = output_grad * np.mean(self.derivative, axis=0)
        input_grad = np.dot(self.weights.T, output_grad)
        weights_grad = np.dot(output_grad, np.mean(self.input, axis=0).T)
        self._weights_grad = weights_grad
        self._biases_grad = output_grad

        return input_grad

    @property
    def weights_grad(self):
        return self._weights_grad

    @property
    def bias_grad(self):
        return self._biases_grad

    def update(self, w_grad, b_grad):
        self.weights -= w_grad
        self.biases -= b_grad
        # print('update: ', w_grad, b_grad)
