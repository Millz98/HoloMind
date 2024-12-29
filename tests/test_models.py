import unittest
import numpy as np
import os 
from holomind.layers import Dense, ReLU
from holomind.optimizers import SGD
from holomind.loss import MeanSquaredError
from holomind.models import Model

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.model.add(Dense(2, 2))  # Input layer
        self.model.add(ReLU())        # Activation layer
        self.model.add(Dense(2, 1))   # Output layer
        self.model.compile(loss_function=MeanSquaredError(), optimizer=SGD(learning_rate=0.01))

        # Sample data
        self.X = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.y = np.array([[1.0], [0.0]])

    def test_fit(self):
        # Train the model for a few epochs
        self.model.fit(self.X, self.y, epochs=1)

        # Check if the model's output shape is correct after training
        output = self.model.layers[-1].forward(self.model.layers[-2]. forward(self.model.layers[0].forward(self.X)))
        self.assertEqual(output.shape, (2, 1))

    def test_loss_function(self):
        # Test the loss function after a forward pass
        output = self.model.layers[-1].forward(self.model.layers[-2].forward(self.model.layers[0].forward(self.X)))
        loss = self.model.loss_function.forward(self.y, output)
        self.assertIsInstance(loss, float)

class TestModelCheckpointing(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.model.add(Dense(2, 2))
        self.model.add(ReLU())
        self.test_filepath = 'test_model.pkl'

    def test_save_load(self):
        # Save the model
        self.model.save(self.test_filepath)

        # Create a new model and load the saved weights
        new_model = Model()
        new_model.add(Dense(2, 2))
        new_model.add(ReLU())
        new_model.load(self.test_filepath)

        # Check if the layers are the same
        for layer, new_layer in zip(self.model.layers, new_model.layers):
            if hasattr(layer, 'weights'):  # Only check layers with weights
                np.testing.assert_almost_equal(layer.weights, new_layer.weights)
                np.testing.assert_almost_equal(layer.biases, new_layer.biases)

    def tearDown(self):
        # Clean up the test file
        if os.path.exists(self.test_filepath):
            os.remove(self.test_filepath)        

if __name__ == '__main__':
    unittest.main()