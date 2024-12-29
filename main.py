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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from holomind.datasets import Dataset





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

    X_train, X_test, y_train, y_test = train_test_split(dataset.drop(non_numerical_columns, axis=1), dataset['ethanol'], test_size=0.2, random_state=42)
    print(y_train.dtype)
    
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

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
        loss = criterion(outputs, torch.from_numpy(y_train).float())
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Visualize the model architecture
    visualize_model_architecture(model)

    # Print a message indicating training is complete
    print("Training complete!")

if __name__ == "__main__":
    main()