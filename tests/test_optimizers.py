import unittest
import numpy as np
from holomind.layers import Dense
from holomind.optimizers import SGD, LearningRateScheduler


class TestSGDOptimizer(unittest.TestCase):
    def setUp(self):
        # Create a Dense layer for testing
        self.input_size = 3
        self.output_size = 2
        self.layer = Dense(self.input_size, self.output_size)
        self.optimizer = SGD(learning_rate=0.01)

        # Initialize weights and biases for testing
        self.layer.weights = np.array([[0.1, 0.2],
                                       [0.3, 0.4],
                                       [0.5, 0.6]])
        self.layer.biases = np.array([[0.1, 0.2]])

    def test_update_weights(self):
        # Simulate gradients
        self.layer.d_weights = np.array([[0.01, 0.02],
                                          [0.03, 0.04],
                                          [0.05, 0.06]])
        self.layer.d_biases = np.array([[0.01, 0.02]])

        # Store original weights and biases
        original_weights = self.layer.weights.copy()
        original_biases = self.layer.biases.copy()

        # Update weights and biases
        self.optimizer.update(self.layer)

        # Check if weights and biases are updated correctly
        expected_weights = original_weights - 0.01 * self.layer.d_weights
        expected_biases = original_biases - 0.01 * self.layer.d_biases

        np.testing.assert_almost_equal(self.layer.weights, expected_weights, decimal=5)
        np.testing.assert_almost_equal(self.layer.biases, expected_biases, decimal=5)

class TestLearningRateScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = LearningRateScheduler(initial_lr=0.1, decay_factor=0.5, epochs_drop=10)

    def test_get_lr(self):
        # Test learning rate at different epochs
        self.assertAlmostEqual(self.scheduler.get_lr(0), 0.1)
        self.assertAlmostEqual(self.scheduler.get_lr(10), 0.05)
        self.assertAlmostEqual(self.scheduler.get_lr(20), 0.025)        

if __name__ == '__main__':
    unittest.main()