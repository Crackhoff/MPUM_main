import numpy as np


class CNN:
    def __init__(self, learning_rate= 0.01, epochs=10, batch_size=32, verbose=1):
        self.layers = []
        
        self.next_layer = 0
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
    def add(self, layer):
        self.layers.append(layer)
        
        layer.set_id(self.next_layer)
        self.next_layer += 1
        
    def train(self, X, y):
        self.batch_learning_rate = self.learning_rate / self.batch_size
        
        self.error = []
        
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                
                self._forward(X_batch)
                self._backward(y_batch)
                
            if self.verbose:
                print("Epoch", epoch, "Error", self.error[-1])
        
    def _forward(self, X):
        self.layers[0].forward(X)
        
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].output)
            
    def _backward(self, y):
        self.layers[-1].backward(y)
        
        for i in range(len(self.layers)-2, -1, -1):
            self.layers[i].backward(self.layers[i+1].delta)
            
        self.error.append(np.mean(np.abs(self.layers[-1].delta)))
        