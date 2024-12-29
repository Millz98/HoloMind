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
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from holomind.datasets import Dataset

class PyTorchDense(nn.Module):
    def __init__(self, input_size, output_size):
        super(PyTorchDense, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.fc1 = PyTorchDense(4, 64)  # Update the input size to 4
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = PyTorchDense(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = PyTorchDense(32, 1)
        self.layers = [
            {"name": "Dense", "input_size": 4, "output_size": 64},
            {"name": "BatchNormalization", "input_size": 64},
            {"name": "ReLU"},
            {"name": "Dropout", "rate": 0.3},
            {"name": "Dense", "input_size": 64, "output_size": 32},
            {"name": "BatchNormalization", "input_size": 32},
            {"name": "ReLU"},
            {"name": "Dropout", "rate": 0.3},
            {"name": "Dense", "input_size": 32, "output_size": 1}
        ]

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def main():
    # Load the dataset
    dataset = pd.read_csv("./holomind/gsalc.csv")

    # Separate numerical columns from non-numerical columns
    numerical_columns = dataset.select_dtypes(include=['int64', 'float64']).columns
    non_numerical_columns = dataset.select_dtypes(exclude=['int64', 'float64']).columns

    # Apply StandardScaler to numerical columns
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataset.drop(non_numerical_columns, axis=1), dataset[non_numerical_columns], test_size=0.2, random_state=42)

    # Create a HoloMind dataset object
    holomind_dataset = Dataset(X_train, y_train)

    # Create a PyTorch model
    model = PyTorchModel()

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the network
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(torch.from_numpy(X_train.values).float())
        loss = criterion(outputs, torch.from_numpy(y_train.values).float())
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Visualize the model architecture
    visualize_model_architecture(model)

    # Print a message indicating training is complete
    print("Training complete!")

if __name__ == "__main__":
    main()