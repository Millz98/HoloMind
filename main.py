# main.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from holomind.models import Model
from holomind.layers import Dense, ReLU, Dropout, BatchNormalization
from holomind.optimizers import SGD, Adam, LearningRateScheduler
from holomind.loss import MeanSquaredError
from holomind.utils import visualize_model_architecture, visualize_performance_metrics

class PyTorchModel(nn.Module):
    def __init__(self, model):
        super(PyTorchModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward(x)

def main():
    # Generate some sample data
    X = np.random.rand(100, 3)  # 100 samples, 3 features
    y = np.random.rand(100, 1)  # 100 target values

    # Create a HoloMind model
    model = Model()

    # Add layers to the model
    model.add(Dense(input_size=3, output_size=64))  # First layer
    model.add(BatchNormalization(input_size=64))  # Added batch normalization after the first layer
    model.add(ReLU())
    model.add(Dropout(rate=0.3))  # Increased dropout rate
    model.add(Dense(input_size=64, output_size=32))  # More layers
    model.add(BatchNormalization(input_size=32))  # Added batch normalization after the second layer
    model.add(ReLU())
    model.add(Dropout(rate=0.3))
    model.add(Dense(input_size=32, output_size=1))  # Output layer

    # Wrap the HoloMind model in a PyTorch model
    pytorch_model = PyTorchModel(model)

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

    # Train the network
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = pytorch_model(torch.from_numpy(X).float())
        loss = criterion(outputs, torch.from_numpy(y).float())
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Visualize the model architecture
    visualize_model_architecture(model)

    # Print a message indicating training is complete
    print("Training complete!")

if __name__ == "__main__":
    main()