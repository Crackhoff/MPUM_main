import numpy as np

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

class ConvLayer(Layer):
    def __init__(self, num_filters: int, filter_size: tuple, stride, padding, input_shape):
        '''
        do we want stride and padding?
        
        do we want input_shape?
        
        do we want to initialize the weights here?
        
        do we want to initialize the biases here?
        
        do we want to initialize the activation here?
        '''
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
    
    def forward(self, X):
        pass
        
    
    def backward(self, delta):
        pass
    
        
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
