# layers.py

import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, d_output):
        self.d_weights = np.dot(self.inputs.T, d_output)
        self.d_biases = np.sum(d_output, axis=0, keepdims=True)
        return np.dot(d_output, self.weights.T)

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, d_output, inputs):
        return d_output * (inputs > 0)
class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self, inputs, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape)
            return inputs * self.mask
        else:
            return inputs * (1 - self.rate)  # Scale the output during inference

    def backward(self, d_output):
        return d_output * self.mask  # Apply the mask to the gradient      