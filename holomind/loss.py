import numpy as np  # Importing NumPy

# holomind/loss.py
class MeanSquaredError:
    @staticmethod
    def forward(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def backward(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size