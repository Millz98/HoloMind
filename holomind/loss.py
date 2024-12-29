import numpy as np  # Importing NumPy

# holomind/loss.py
class MeanSquaredError:
    def forward(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def backward(self, y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size  # Gradient of MSE