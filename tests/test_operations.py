import unittest
import numpy as np
from holomind.operations import Addition, Subtraction, ElementwiseMultiply, Sigmoid, Tanh, MatrixMultiply

class TestOperations(unittest.TestCase):

    def test_matrix_multiply(self):
        a = MatrixMultiply(np.array([[1, 2], [3, 4]]), np.array([[5], [6]]))
        result = a.forward()
        expected = np.array([[17], [39]])
        np.testing.assert_array_equal(result, expected)

    def test_addition(self):
        a = Addition(np.array([1, 2]), np.array([3, 4]))
        result = a.forward()
        expected = np.array([4, 6])
        np.testing.assert_array_equal(result, expected)

    def test_subtraction(self):
        a = Subtraction(np.array([5, 6]), np.array([3, 4]))
        result = a.forward()
        expected = np.array([2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_elementwise_multiply(self):
        a = ElementwiseMultiply(np.array([1, 2]), np.array([3, 4]))
        result = a.forward()
        expected = np.array([3, 8])
        np.testing.assert_array_equal(result, expected)

    def test_sigmoid(self):
        a = Sigmoid(np.array([0, 1, -1]))
        result = a.forward()
        expected = 1 / (1 + np.exp(-np.array([0, 1, -1])))
        np.testing.assert_array_almost_equal(result, expected)

    def test_tanh(self):
        a = Tanh(np.array([0, 1, -1]))
        result = a.forward()
        expected = np.tanh(np.array([0, 1, -1]))
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == '__main__':
    unittest.main()