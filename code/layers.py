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
    def weights(self):
        pass

    @property
    def bias(self):
        pass

class ConvLayer(Layer):
    def __init__(self, num_kernels: int, kernel_size: int, input_shape: tuple, learning_rate=0.01):
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

        self.kernels_grad = np.zeros(self.kernels_shape)
        self.biases_grad = np.zeros(self.output_shape)

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
        return self.kernels

    @property
    def bias(self):
        return self.biases

class PoolLayer(Layer):
    def __init__(self, num_filters: int, filter_size: tuple, stride, padding, input_shape):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
    
        
class MaxPoolLayer(Layer):
    def __init__(self, pool_size, input_shape, stride = None, pool_type='max'):
        self.pool_size = pool_size
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride
        self.input_shape = input_shape
        self.pool_type = pool_type
        self.output_shape = (input_shape[0], (input_shape[1] - self.pool_size[0]) // self.stride[0] + 1, (input_shape[2] - self.pool_size[1]) // self.stride[1] + 1, input_shape[3])
        
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
                    self.output[:, i // self.stride[0], j // self.stride[1], c] = np.max(X[:, i:i+self.pool_size[0], j:j+self.pool_size[1], c], axis=(1, 2))
                    self.indices[:, i:i+self.pool_size[0], j:j+self.pool_size[1], c] = np.where(X[:, i:i+self.pool_size[0], j:j+self.pool_size[1], c] == self.output[:, i // self.stride[0], j // self.stride[1], c].reshape(-1, 1, 1), 1, 0)
                    
        self.indices = self._compress_indices()
        print(self.indices.shape)

        return self.output
    
    def backward(self, delta):
        # implement backpropagation for max pooling using the saved indices
        
        # delta is in the form (x.output, y.output, channel)
        # we want to return delta in the form (x.input, y.input, channel)
                
        # for each pool, we want to split the delta into the pool size
        
        delta_out = np.zeros(self.input_shape)
        
        # duplicate every delta value pool_size[0] times along axis 1
        
        delta = np.repeat(delta, self.pool_size[0], axis=0)
        
        delta = np.repeat(delta, self.pool_size[1], axis=1)
        
        delta_out = delta * self.indices
        
        return delta_out
    
class MeanPoolLayer(Layer):
    def __init__(self, pool_size, input_shape, stride = None, pool_type='max'):
        self.pool_size = pool_size
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride
        self.input_shape = input_shape
        self.pool_type = pool_type
        
        self.output_shape = (input_shape[0], (input_shape[1] - self.pool_size[0]) // self.stride[0] + 1, (input_shape[2] - self.pool_size[1]) // self.stride[1] + 1, input_shape[3])
    
    def forward(self, X):
        # implement mean pooling
        self.output = np.zeros(self.output_shape)        
        
        reshaped_input = X.reshape(X.shape[0], X.shape[1] // self.pool_size[0], self.pool_size[0], X.shape[2] // self.pool_size[1], self.pool_size[1], X.shape[3])
        output = reshaped_input.mean(axis=(2, 4))
        reshaped_input = reshaped_input.transpose(0, 1, 3, 5, 2, 4)
        # self.indices = each neuron equal weight
        return self.activate(output)        
        
                        
    
    def backward(self, delta):
        # implement backpropagation for mean pooling
        
        # delta is in the form (batch, x.output, y.output, channel)
        # reshape delta to (batch, x.output, y.output, 1)
        # repeat delta pool_size[0] times along axis 3
        # repeat delta pool_size[1] times along axis 4

        delta = np.repeat(delta, self.pool_size[0], axis=0)
        delta = np.repeat(delta, self.pool_size[1], axis=1)
        
        # divide delta by the pool size
        delta = delta / (self.pool_size[0] * self.pool_size[1])

        return delta
    
class FlattenLayer(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], np.prod(input_shape[1:]))
        
    def forward(self, X):
        self.input = X
        self.output = X.reshape(X.shape[0], -1)
        
        return self.output
        
    def backward(self, delta):
        return delta.reshape(self.input_shape)

class DenseLayer(Layer):
    def __init__(self, units, input_shape, activation='relu', learning_rate=0.001):
        self.units = units
        self.input_shape = input_shape
        self.activation = activation
        self.learning_rate = learning_rate
        
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
