import unittest
import numpy as np
from holomind.loss import MeanSquaredError

class TestMeanSquaredError(unittest.TestCase):
    def setUp(self):
        self.loss = MeanSquaredError()

    def test_forward(self):
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.0], [2.0], [2.0]])
        expected_loss = np.mean((y_true - y_pred) ** 2)

        loss_value = self.loss.forward(y_true, y_pred)
        self.assertAlmostEqual(loss_value, expected_loss, places=5)

    def test_backward(self):
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.0], [2.0], [2.0]])
        expected_gradient = 2 * (y_pred - y_true) / y_true.size

        gradient = self.loss.backward(y_true, y_pred)
        np.testing.assert_almost_equal(gradient, expected_gradient, decimal=5)

if __name__ == '__main__':
    unittest.main()