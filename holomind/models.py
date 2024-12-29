# models.py

from holomind.layers import Dense, ReLU  # Import the necessary layers
from holomind.loss import MeanSquaredError
from holomind.optimizers import SGD
import pickle

class Model:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.inputs = []  # Store inputs for backward pass

    def add(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)

    def compile(self, loss_function, optimizer):
        """Compile the model with a loss function and optimizer."""
        self.loss_function = loss_function
        self.optimizer = optimizer

    def save(self, filepath):
        """Save the model's layers to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.layers, f)

    def load(self, filepath):
        """Load the model's layers from a file."""
        with open(filepath, 'rb') as f:
            self.layers = pickle.load(f)

    def fit(self, X, y, epochs):
        """Train the model for a specified number of epochs."""
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            gradient_output = self.backward_pass(y, output)
            self.update_weights()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

    def compute_loss(self, y, output):
        """Compute the loss for the current output."""
        return self.loss_function.forward(y, output)

    def forward(self, X):
        """Perform a forward pass through the model."""
        self.inputs = []  # Store inputs for backward pass
        for layer in self.layers:
            self.inputs.append(X)  # Store the input to the layer
            X = layer.forward(X)
        return X

    def backward_pass(self, y, output):
        """Perform the backward pass and return the gradient."""
        gradient_output = self.loss_function.backward(y, output)
        self.backward(gradient_output)

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