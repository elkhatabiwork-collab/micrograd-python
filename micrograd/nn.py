import numpy as np
from micrograd.engine import Value

rng = np.random.default_rng(seed=101)

class Neuron:
    def __init__(self, n_inputs):
        self.weights = [Value(rng.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(0.0)
    
    def __call__(self, x):
        act = sum(xi * wi for xi, wi in zip(x, self.weights)) + self.bias
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.weights + [self.bias]
    

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    #forward 
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, n_inputs, n_outputs):
        sizes = [n_inputs] + n_outputs
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


        