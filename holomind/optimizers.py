# optimizers.py

import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layers):
        """Update the weights of the layers."""
        if not isinstance(layers, list):
            layers = [layers]  # Convert to a list if a single layer is passed

        for layer in layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * layer.d_weights
                layer.biases -= self.learning_rate * layer.d_biases


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []  # First moment vector
        self.v = []  # Second moment vector
        self.t = 0  # Time step

    def initialize_moments(self, layers):
        for layer in layers:
            if hasattr(layer, 'weights'):
                self.m.append(np.zeros_like(layer.weights))
                self.v.append(np.zeros_like(layer.weights))
            if hasattr(layer, 'biases'):
                self.m.append(np.zeros_like(layer.biases))
                self.v.append(np.zeros_like(layer.biases))

    def update(self, layers):
        """Update the weights of the layers using Adam optimization."""
        if not isinstance(layers, list):
            layers = [layers]  # Convert to a list if a single layer is passed

        if not self.m or not self.v:
            self.initialize_moments(layers)

        self.t += 1

        moment_index = 0
        for layer in layers:
            if hasattr(layer, 'weights'):
                # Update biased first moment estimate
                self.m[moment_index] = self.beta1 * self.m[moment_index] + (1 - self.beta1) * layer.d_weights
                # Update biased second moment estimate
                self.v[moment_index] = self.beta2 * self.v[moment_index] + (1 - self.beta2) * (layer.d_weights ** 2)

                # Compute bias-corrected first moment estimate
                m_hat = self.m[moment_index] / (1 - self.beta1 ** self.t)
                # Compute bias-corrected second moment estimate
                v_hat = self.v[moment_index] / (1 - self.beta2 ** self.t)

                # Update weights
                layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

                moment_index += 1

            if hasattr(layer, 'biases'):
                # Update biased first moment estimate
                self.m[moment_index] = self.beta1 * self.m[moment_index] + (1 - self.beta1) * np.sum(layer.d_biases, axis=0, keepdims=True)
                # Update biased second moment estimate
                self.v[moment_index] = self.beta2 * self.v[moment_index] + (1 - self.beta2) * (np.sum(layer.d_biases ** 2, axis=0, keepdims=True))

                # Compute bias-corrected first moment estimate
                m_hat = self.m[moment_index] / (1 - self.beta1 ** self.t)
                # Compute bias-corrected second moment estimate
                v_hat = self.v[moment_index] / (1 - self.beta2 ** self.t)

                # Update biases
                layer.biases -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

                moment_index += 1


class LearningRateScheduler:
    def __init__(self, initial_lr, decay_factor, epochs_drop):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.epochs_drop = epochs_drop

    def get_lr(self, epoch):
        return self.initial_lr * (self.decay_factor ** (epoch // self.epochs_drop))