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
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0  # Time step

    def update(self, layers):
        """Update the weights of the layers using Adam optimization."""
        if not isinstance(layers, list):
            layers = [layers]  # Convert to a list if a single layer is passed

        if self.m is None:
            self.m = [np.zeros_like(layer.weights) for layer in layers if hasattr(layer, 'weights')]
            self.v = [np.zeros_like(layer.weights) for layer in layers if hasattr(layer, 'weights')]

        self.t += 1

        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights'):
                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * layer.d_weights
                # Update biased second moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (layer.d_weights ** 2)

                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                # Compute bias-corrected second moment estimate
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Print shapes for debugging
                print(f"Layer {i}: m_hat shape: {m_hat.shape}, v_hat shape: {v_hat.shape}, d_weights shape: {layer.d_weights.shape}")

                # Update weights
                layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

                # Update biases
                print(f"Updating biases: layer.biases shape: {layer.biases.shape}, layer.d_biases shape: {layer.d_biases.shape}")
                layer.biases -= self.learning_rate * layer.d_biases / (np.sqrt(v_hat) + self.epsilon)


class LearningRateScheduler:
    def __init__(self, initial_lr, decay_factor, epochs_drop):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.epochs_drop = epochs_drop

    def get_lr(self, epoch):
        return self.initial_lr * (self.decay_factor ** (epoch // self.epochs_drop))