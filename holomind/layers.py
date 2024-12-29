# layers.py

import numpy as np
from holomind.operations import MatrixMultiply

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, d_output):
        print(f"Backward pass d_output shape: {d_output.shape}")  # Debugging
        if self.inputs is None:
            raise ValueError("Inputs must be set before calling backward.")
        
        self.d_weights = np.dot(self.inputs.T, d_output)  # Shape: (input_size, output_size)
        self.d_biases = np.sum(d_output, axis=0, keepdims=True)  # Shape: (1, output_size)
        return np.dot(d_output, self.weights.T)  # Shape: (batch_size, input_size)

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs  # Store inputs for backward pass
        return np.maximum(0, inputs)  # Apply ReLU activation

    def backward(self, d_output):
        return d_output * (self.inputs > 0)  # Gradient of ReLU

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
        if self.mask is None:
            raise ValueError("Mask must be set before calling backward.")
        return d_output * self.mask  # Apply the mask to the gradient

class BatchNormalization:
    def __init__(self, input_size, momentum=0.9, epsilon=1e-5):
        self.input_size = input_size
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = np.ones((1, input_size))  # Scale parameter
        self.beta = np.zeros((1, input_size))  # Shift parameter
        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))
        self.training = True  # Flag to indicate training or inference mode
        self.inputs = None  # Initialize inputs for backward pass

    def forward(self, inputs):
        self.inputs = inputs  # Store inputs for backward pass
        if self.training:
            # Calculate mean and variance
            self.batch_mean = np.mean(inputs, axis=0, keepdims=True)
            self.batch_var = np.var(inputs, axis=0, keepdims=True)

            # Normalize the inputs
            self.x_normalized = (inputs - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var

            # Scale and shift
            return self.gamma * self.x_normalized + self.beta
        else:
            # Use running mean and variance for inference
            x_normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * x_normalized + self.beta

    def backward(self, d_output):
        N, D = d_output.shape

        # Compute gradients for gamma and beta
        self.d_gamma = np.sum(d_output * self.x_normalized, axis=0)
        self.d_beta = np.sum(d_output, axis=0)

        # Compute gradient for the normalized input
        dx_normalized = d_output * self.gamma

        # Compute gradients for the batch mean and variance
        d_var = np.sum(dx_normalized * (self.inputs - self.batch_mean) * -0.5 * np.power(self.batch_var + self.epsilon, -1.5), axis=0)
        d_mean = np.sum(dx_normalized * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0) + d_var * np.mean(-2 * (self.inputs - self.batch_mean), axis=0)

        # Compute gradient for the input
        d_input = (dx_normalized / np.sqrt(self.batch_var + self.epsilon)) + (d_var * 2 * (self.inputs - self.batch_mean) / N) + (d_mean / N)

        return d_input  # Return the gradient for the input
