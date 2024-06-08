import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size, stride, padding):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, X):
        pass
        
    
    def backward(self, delta):
        pass
    
        
class PoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, X):
        pass
    
    def backward(self, delta):
        pass
    
class FlattenLayer:
    def forward(self, X):
        pass
    
    def backward(self, delta):
        pass
    
class DenseLayer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
    
    def forward(self, X):
        pass
    
    def backward(self, delta):
        pass
    
class ActivationLayer:
    def __init__(self, activation):
        self.activation = activation
    
    def forward(self, X):
        pass
    
    def backward(self, delta):
        pass
