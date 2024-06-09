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
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
    
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
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
    
    def forward(self, X):
        pass
    
    def backward(self, delta):
        pass
