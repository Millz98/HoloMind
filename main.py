# main.py
from holomind.models import PyTorchModel
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from holomind.datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset

class PyTorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PyTorchModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)  # Output raw logits

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # No softmax here
        return x

def main():
    # Load the dataset
    dataset = pd.read_csv("./holomind/gsalc.csv")

    # Separate numerical columns from non-numerical columns
    numerical_columns = dataset.select_dtypes(include=['int64', 'float64']).columns
    non_numerical_columns = dataset.select_dtypes(exclude=['int64', 'float64']).columns

    # Apply StandardScaler to numerical columns
    scaler = StandardScaler()
    dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

    # Split the dataset into features and target variable
    X = dataset.drop(non_numerical_columns, axis=1)
    y = dataset['ethanol']  # Assuming 'ethanol' is the target variable

    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train.values).float()
    y_train_tensor = torch.from_numpy(y_train).long()  # Use long for class labels
    X_val_tensor = torch.from_numpy(X_val.values).float()
    y_val_tensor = torch.from_numpy(y_val).long()  # Use long for class labels

    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create a PyTorch model
    input_size = X_train.shape[1]  # Number of features
    num_classes = len(le.classes_)  # Number of unique classes
    model = PyTorchModel(input_size, num_classes)

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define optimizer here

    # Train the network
    for epoch in range(50):
        model.train()  # Set the model to training mode
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}, Validation Loss: { val_loss.item()}')  # Print average loss for the epoch and validation loss

if __name__ == "__main__":
    main()