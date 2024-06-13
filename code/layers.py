import numpy as np
from scipy import signal


class Layer:
    def __init__(self, input_shape: tuple):
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

    @property
    def weights_shape(self):
        return ()

    @property
    def bias_shape(self):
        return ()

    def activate(self, X):
        if self.activation == 'relu':
            return np.maximum(X, 0)
        elif self.activation == 'sigmoid':
            pos = np.where(X > 0)
            neg = np.where(X <= 0)
            res = np.zeros(X.shape)
            res[pos] = 1 / (1 + np.exp(-X[pos]))
            res[neg] = np.exp(X[neg]) / (1 + np.exp(X[neg]))
            return res
        # elif self.activation == 'softmax':
        #     exps = np.exp(X - np.max(X))
        #     return exps / np.sum(exps)
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
            sX = self.activate(X)
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
            for i in range(self.num_kernels):
                for j in range(self.num_channels):
                    self.output[b, i] += signal.correlate2d(self.input[b, j], self.kernels[i, j], mode='valid')

        self.derivative = self.activate_derivative(self.output)
        return self.activate(self.output)

    def backward(self, output_grad):
        batches_num = self.input.shape[0]
        output_grad = output_grad * np.mean(self.derivative, axis=0)
        kernels_grad = np.zeros((batches_num, *self.kernels_shape))  # (K, C, KS, KS)
        input_grad = np.zeros(self.input_shape)  # (H, W, C)

        for i in range(self.num_kernels):
            for j in range(self.num_channels):
                for b in range(batches_num):
                    kernels_grad[b, i, j] = signal.correlate2d(self.input[b, j], output_grad[i], 'valid')
                input_grad[j] += signal.convolve2d(output_grad[i], self.kernels[i, j], 'full')

        self.kernels_grad = np.mean(kernels_grad, axis=0)
        self.biases_grad = output_grad
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

    def __str__(self) -> str:
        return f'ConvLayer: {self.input_shape} -> {self.output_shape}'

    @property
    def weights_shape(self):
        return self.kernels_shape

    @property
    def bias_shape(self):
        return self.output_shape

    def __str__(self) -> str:
        return f'ConvLayer: {self.input_shape} -> {self.output_shape} with {self.num_kernels} kernels of size {self.kernels_shape[0]}x{self.kernels_shape[1]} \n Activation: {self.activation}'


class MaxPoolLayer(Layer):
    def __init__(self, input_shape: tuple, pool_size: tuple, stride=None):
        self.pool_size = pool_size
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], (input_shape[1] - self.pool_size[0]) // self.stride[0] + 1,
                             (input_shape[2] - self.pool_size[1]) // self.stride[1] + 1)  # (C, H, W)

    def _compress_indices(self):
        return np.mean(self.indices, axis=0)

    def forward(self, X):
        # implement max pooling and save the indices of the max values
        (batch, channel, x, y) = X.shape
        # add padding if needed
        self.output = np.zeros((batch, *self.output_shape))

        self.indices = np.zeros(X.shape)
        for b in range(batch):
            for c in range(channel):
                for i in range(0, x - self.pool_size[0] + 1, self.stride[0]):
                    for j in range(0, y - self.pool_size[1] + 1, self.stride[1]):
                        self.output[b, c, i // self.stride[0], j // self.stride[1]] = np.max(
                            X[b, c, i:i + self.pool_size[0], j:j + self.pool_size[1]])
                        self.indices[b, c, i:i + self.pool_size[0], j:j + self.pool_size[1]] = np.where(
                            X[b, c, i:i + self.pool_size[0], j:j + self.pool_size[1]] == self.output[
                                b, c, i // self.stride[0],
                                      j // self.stride[1],
                            ].reshape(-1, 1, 1), 1, 0)

        self.indices = self._compress_indices()

        return self.output

    def backward(self, output_grad):
        # implement backpropagation for max pooling using the saved indices
        output_grad = np.repeat(output_grad, self.pool_size[0], axis=1)

        output_grad = np.repeat(output_grad, self.pool_size[1], axis=2)

        if output_grad.shape[1] != self.input_shape[1]:
            output_grad = np.pad(output_grad, ((0, 0), (0, self.input_shape[1] - output_grad.shape[1]), (0, 0)))

        if output_grad.shape[2] != self.input_shape[2]:
            output_grad = np.pad(output_grad, ((0, 0), (0, 0), (0, self.input_shape[2] - output_grad.shape[2])))

        output_grad = output_grad * self.indices

        return output_grad

    def __str__(self) -> str:
        return f'MaxPoolLayer: {self.input_shape} -> {self.output_shape} with pool size {self.pool_size}'


class MeanPoolLayer(Layer):
    def __init__(self, input_shape: tuple, pool_size: tuple, stride=None):
        self.pool_size = pool_size
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], (input_shape[1] - self.pool_size[0]) // self.stride[0] + 1,
                             (input_shape[2] - self.pool_size[1]) // self.stride[1] + 1)  # (C, H, W)

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

        if output_grad.shape[1] != self.input_shape[1]:
            output_grad = np.pad(output_grad, ((0, 0), (0, self.input_shape[1] - output_grad.shape[1]), (0, 0)))

        if output_grad.shape[2] != self.input_shape[2]:
            output_grad = np.pad(output_grad, ((0, 0), (0, 0), (0, self.input_shape[2] - output_grad.shape[2])))

        return output_grad

    def __str__(self) -> str:
        return f'MeanPoolLayer: {self.input_shape} -> {self.output_shape} with pool size {self.pool_size}'


class FlattenLayer(Layer):
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], np.prod(input_shape[1:]))

    def forward(self, X):
        self.input = X
        self.output = X.reshape(X.shape[0], -1, 1)
        return self.output

    def backward(self, output_grad):
        return output_grad.reshape(self.input_shape)

    def __str__(self) -> str:
        return f'FlattenLayer: {self.input_shape} -> {self.output_shape}'


class DenseLayer(Layer):
    def __init__(self, input_size: tuple, output_size: int, activation='relu'):
        self.input_size = input_size[0]
        self.output_size = output_size
        self.activation = activation

        self.weights = np.random.randn(output_size, self.input_size)
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

    def __str__(self) -> str:
        return f'DenseLayer: {self.input_size} -> {self.output_size}'

    @property
    def weights_shape(self):
        return self.weights.shape

    @property
    def bias_shape(self):
        return self.biases.shape


class BatchNormLayer(Layer):
    def __init__(self, input_shape: tuple, epsilon=1e-5):
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.gamma = np.random.randn(input_shape[0], 1)
        self.beta = np.random.randn(input_shape[0], 1)

        self._gamma_grad = None
        self._beta_grad = None

    def forward(self, X):
        self.input = X
        self.mean = np.mean(X, axis=0)
        self.variance = np.var(X, axis=0)
        self.norm = (X - self.mean) / np.sqrt(self.variance + self.epsilon)
        self.output = self.gamma * self.norm + self.beta
        return self.output

    def backward(self, output_grad):
        batches_num = self.input.shape[0]
        self._gamma_grad = np.sum(output_grad * self.norm, axis=0)
        self._beta_grad = np.sum(output_grad, axis=0)
        norm_grad = output_grad * self.gamma
        variance_grad = np.sum(norm_grad * (self.input - self.mean), axis=0) * -0.5 * (self.variance + self.epsilon) ** -1.5
        mean_grad = np.sum(norm_grad * -1 / np.sqrt(self.variance + self.epsilon), axis=0) + variance_grad * np.sum(-2 * (self.input - self.mean), axis=0) / batches_num
        input_grad = norm_grad / np.sqrt(self.variance + self.epsilon) + variance_grad * 2 * (self.input - self.mean) / batches_num + mean_grad / batches_num
        return input_grad

    @property
    def weights_grad(self):
        return self._gamma_grad

    @property
    def bias_grad(self):
        return self._beta_grad

    def update(self, w_grad, b_grad):
        self.gamma -= w_grad
        self.beta -= b_grad

    def __str__(self) -> str:
        return f'BatchNormalizationLayer: {self.input_shape}'

    @property
    def weights_shape(self):
        return self.gamma.shape

    @property
    def bias_shape(self):
        return self.beta.shape