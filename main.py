# Import necessary libraries
import numpy as np  # For numerical operations
import torch  # For deep learning
import torch.nn as nn  # For neural network layers
import torch.optim as optim  # For optimization algorithms
from torch.utils.data import DataLoader, TensorDataset  # For handling data
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For data preprocessing
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # For model evaluation
from sklearn.utils import class_weight  # For handling imbalanced classes
from imblearn.over_sampling import SMOTE  # For balancing imbalanced data
import pandas as pd  # For data manipulation
import os  # For file operations
import logging  # For logging information
import json  # For reading configuration

# Set up logging to show information during execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="config.json"):
    """Loads settings from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def validate_data(file_path):
    """Checks the data for any issues like missing values or duplicates."""
    try:
        df = pd.read_csv(file_path)

        # Log information about the data
        logging.info("--- Data Validation ---")
        logging.info(f"Missing Values:\n{df.isnull().sum()}")
        logging.info(f"Data Types:\n{df.dtypes}")
        logging.info(f"Duplicate Rows: {df.duplicated().sum()}")
        logging.info(f"Descriptive Statistics:\n{df.describe()}")

        # Warn if there are any problems
        if df.isnull().sum().sum() > 0:
            logging.warning("Warning: Missing values found in the data.")
        if df.duplicated().sum() > 0:
            logging.warning('Warning: Duplicate rows found in the data.')

        return df

    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        raise
    except pd.errors.ParserError:
        logging.error("Error: CSV parsing failed. Check for formatting issues.")
        raise

def load_and_preprocess_data(config):
    """Prepares the data for training by cleaning and transforming it."""
    try:
        # Load and validate the data
        logging.info(f"Loading data from: {config['data_path']}")
        dataset = validate_data(config['data_path'])
        logging.info("Data validation successful.")

        # Convert categorical variables to numbers
        dataset = pd.get_dummies(dataset, columns=['100ppb'])
        le = LabelEncoder()
        dataset[config["target_column"]] = le.fit_transform(dataset[config["target_column"]])

    except Exception as e:
        logging.critical(f"An error occurred during data validation: {e}")
        raise

    # Scale numerical features to have similar ranges
    numerical_columns = dataset.select_dtypes(include=['int64', 'float64']).columns
    numerical_columns = numerical_columns.drop(labels=config["target_column"], errors='ignore')
    scaler = StandardScaler()
    dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

    # Split data into features (X) and target (y)
    X = dataset.drop(columns=config["target_column"])
    y = dataset[config["target_column"]]

    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"], stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=config["random_state"], stratify=y_temp)

    # Print class distribution
    print(pd.Series(y_train).value_counts())

    # Balance the training data using SMOTE
    smote = SMOTE(random_state=config["random_state"])
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Convert data to PyTorch tensors
    X_train_resampled = X_train_resampled.astype(float)
    X_val = X_val.astype(float)
    X_test = X_test.astype(float)

    X_train_tensor = torch.from_numpy(X_train_resampled.values).float()
    y_train_tensor = torch.from_numpy(y_train_resampled.values).long()
    X_val_tensor = torch.from_numpy(X_val.values).float()
    y_val_tensor = torch.from_numpy(y_val.values).long()
    X_test_tensor = torch.from_numpy(X_test.values).float()
    y_test_tensor = torch.from_numpy(y_test.values).long()

    # Create data loaders for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    return train_loader, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, le, dataset

def create_model(input_size, num_classes, config):
    """Creates a neural network model with the specified architecture."""
    model = nn.Sequential(
        nn.Linear(input_size, config["hidden_size"]),  # Input layer
        nn.BatchNorm1d(config["hidden_size"]),  # Normalize the data
        nn.ReLU(),  # Activation function
        nn.Dropout(config["dropout_rate"]),  # Prevent overfitting
        nn.Linear(config["hidden_size"], num_classes)  # Output layer
    )
    return model

def train_model(model, train_loader, X_val_tensor, y_val_tensor, config, y_train_resampled):
    """Trains the neural network model."""
    # Calculate class weights to handle imbalanced data
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Set up optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0

        # Train on batches
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip"])
            optimizer.step()
            epoch_loss += loss.item()

            if config["debug_level"] == 2:
                logging.info(f"Epoch: {epoch + 1}, Batch Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        scheduler.step()

        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config["model_save_path"])
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["early_stopping_patience"]:
                logging.info("Early stopping triggered")
                break

        logging.info(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}, Validation Loss: {val_loss.item()}')

def evaluate_model(model, X_test_tensor, y_test_tensor, le):
    """Evaluates the trained model on the test set."""
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = nn.CrossEntropyLoss()(test_outputs, y_test_tensor)
        predicted = torch.argmax(test_outputs, dim=1).numpy()
        actual = y_test_tensor.numpy()

        # Calculate various performance metrics
        accuracy = accuracy_score(actual, predicted)
        precision = precision_score(actual, predicted, average='weighted', zero_division=0)
        recall = recall_score(actual, predicted, average='weighted', zero_division=0)
        f1 = f1_score(actual, predicted, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(actual, predicted)

        # Log the results
        logging.info(f'Test Loss: {test_loss.item()}')
        logging.info(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
        logging.info(f'Confusion Matrix:\n{conf_matrix}')

def main():
    """Main function that runs the entire pipeline."""
    # Load configuration and prepare data
    config = load_config()
    train_loader, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, le, dataset = load_and_preprocess_data(config)
    
    # Create and train the model
    input_size = X_val_tensor.shape[1]
    num_classes = len(le.classes_)
    model = create_model(input_size, num_classes, config)
    y_train_resampled = train_loader.dataset.tensors[1].numpy()
    train_model(model, train_loader, X_val_tensor, y_val_tensor, config, y_train_resampled)
    
    # Load the best model and evaluate it
    model.load_state_dict(torch.load(config["model_save_path"], weights_only=True))
    evaluate_model(model, X_test_tensor, y_test_tensor, le)

if __name__ == "__main__":
    main()