import unittest
import numpy as np
from holomind.layers import Dense, Dropout

class TestDenseLayer(unittest.TestCase):
    def setUp(self):
        # Set up a Dense layer for testing
        self.input_size = 3
        self.output_size = 2
        self.layer = Dense(self.input_size, self.output_size)

        # Create a sample input
        self.inputs = np.array([[1.0, 2.0, 3.0], 
                                [4.0, 5.0, 6.0]])

        # Forward pass to set inputs
        self.layer.forward(self.inputs)

    def test_forward(self):
        # Test the forward pass
        output = self.layer.forward(self.inputs)
        expected_output = np.dot(self.inputs, self.layer.weights) + self.layer.biases
        np.testing.assert_almost_equal(output, expected_output, decimal=5)

    def test_backward(self):
        # Test the backward pass
        d_output = np.array([[1.0, 0.5], 
                             [0.5, 1.0]])  # Sample gradient from the next layer
        d_input = self.layer.backward(d_output)

        # Check the shape of the gradient with respect to the input
        self.assertEqual(d_input.shape, self.inputs.shape)

        # Check the gradients of weights and biases
        self.assertEqual(self.layer.d_weights.shape, (self.input_size, self.output_size))
        self.assertEqual(self.layer.d_biases.shape, (1, self.output_size))

class TestDropoutLayer(unittest.TestCase):
    def setUp(self):
        self.dropout = Dropout(rate=0.5)
        np.random.seed(42) 

    def test_forward_training(self):
        inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
        output = self.dropout.forward(inputs, training=True)
        # Check that the output has the same shape as the input
        self.assertEqual(output.shape, inputs.shape)
        # Check that some values are zeroed out
        self.assertTrue(np.any(output == 0))

    def test_forward_inference(self):
        inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
        output = self.dropout.forward(inputs, training=False)
        # During inference, the output should be scaled
        expected_output = inputs * (1 - self.dropout.rate)
        np.testing.assert_almost_equal(output, expected_output)

    def test_backward(self):
        d_output = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.dropout.mask = np.array([[1, 0], [1, 1]])  # Simulate the mask
        d_input = self.dropout.backward(d_output)
        expected_d_input = d_output * self.dropout.mask
        np.testing.assert_almost_equal(d_input, expected_d_input)       

if __name__ == '__main__':
    unittest.main()