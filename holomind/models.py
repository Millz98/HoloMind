# models.py
import numpy as np
from holomind.layers import Dense, ReLU  # Import the necessary layers
from holomind.loss import MeanSquaredError
from holomind.optimizers import SGD
from holomind.operations import MatrixMultiply
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configure logging

class Model:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.inputs = []  # Store inputs for backward pass
        self.history = {'loss': [], 'accuracy': []}  # Store training history
    
    def add(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)

    def compile(self, loss_function, optimizer):
        """Compile the model with a loss function and optimizer."""
        self.loss_function = loss_function
        self.optimizer = optimizer

    def fit(self, X, y, epochs):
        """Train the model for a specified number of epochs."""
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            gradient_output = self.backward_pass(y, output)
            self.update_weights()
            self.history['loss'].append(loss)  # Store the loss
            # self.history['accuracy'].append(accuracy)  # Store the accuracy (if applicable)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

    def compute_loss(self, y, output):
        """Compute the loss for the current output."""
        return self.loss_function.forward(y, output)

    def forward(self, X):
        """Perform a forward pass through the model."""
        self.inputs = []  # Store inputs for backward pass
        for layer in self.layers:
            self.inputs.append(X)  # Store the input to the layer
            X = layer.forward(X)  # This now returns an Operation object
        return X  # Return the final operation object

    def backward_pass(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Perform a backward pass through the model to compute gradients.

        Parameters:
        - y_true: np.ndarray
            Ground truth labels.
        - y_pred: np.ndarray
            Predicted labels.

        Returns:
        - loss_gradient: np.ndarray
            Gradient of the loss with respect to the predictions.
        """
        # Compute the gradient of the loss with respect to the predictions
        loss_gradient = self.loss_function.backward(y_true, y_pred)
        print(f"Initial loss gradient: {loss_gradient.shape}")  # Debugging

        # Backpropagate through the layers in reverse order
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)
            print(f"Gradient after {layer.__class__.__name__}: {loss_gradient.shape}")  # Debugging

        return loss_gradient

    def backward(self, gradient_output):
        """Perform a backward pass through the model."""
        for layer, input_data in zip(reversed(self.layers), reversed(self.inputs)):
            if isinstance(layer, ReLU):
                gradient_output = layer.backward(gradient_output, input_data)  # Pass inputs to ReLU
            else:
                gradient_output = layer.backward(gradient_output)

    def update_weights(self):
        """Update the weights of the model."""
        self.optimizer.update(self.layers)

    def save(self, filepath):
        """Save the model's layers to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.layers, f)    


    def load(self, filepath):
        """Load the model's layers from a file."""
        with open(filepath, 'rb') as f:
            self.layers = pickle.load(f)        