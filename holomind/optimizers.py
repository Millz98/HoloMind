# optimizers.py

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


class LearningRateScheduler:
    def __init__(self, initial_lr, decay_factor, epochs_drop):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.epochs_drop = epochs_drop

    def get_lr(self, epoch):
        return self.initial_lr * (self.decay_factor ** (epoch // self.epochs_drop))        