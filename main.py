import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Get the value of the 'debug' environment variable
debug_level = os.getenv('debug')

# Check the value and print corresponding messages
if debug_level == '2':
    print("Debug level is set to 2: Verbose logging enabled.")
    # You can add more verbose logging or debugging information here
elif debug_level == '1':
    print("Debug level is set to 1: Standard logging.")
    # You can add standard logging information here
else:
    print("Debug level is not set or is set to 0: No debug information.")
    # Normal operation without debug information

# Define your model with Batch Normalization and Dropout
class YourModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(YourModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
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

    # Initialize model, criterion, optimizer, and scheduler
    input_size = X_train.shape[1]  # Number of features
    num_classes = len(le.classes_)  # Number of unique classes
    model = YourModel(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Training loop with early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_patience = 5
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            epoch_loss += loss.item()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        # Step the scheduler
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the model if needed
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping triggered")
                break

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}, Validation Loss: {val_loss.item()}')

    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.from_numpy(X_test.values).float())
        test_loss = criterion(test_outputs, torch.from_numpy(y_test).long())
        print(f'Test Loss: {test_loss.item()}')

if __name__ == "__main__":
    main()