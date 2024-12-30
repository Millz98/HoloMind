import numpy as np
import logging
import matplotlib.pyplot as plt
import networkx as nx

from holomind.models import PyTorchModel


def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def visualize_performance(metrics):
    # Plot performance metrics
    plt.plot(metrics['accuracy'])
    plt.plot(metrics['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Performance')
    plt.legend(['Accuracy', 'Loss'])
    plt.show()


def visualize_model_architecture(model):
    # Create a new directed graph
    graph = nx.DiGraph()

    # Add nodes for each layer
    for i, layer in enumerate(model.layers):
        graph.add_node(f"Layer {i+1}: {layer['name']}")

    # Add edges to represent connections between layers
    for i in range(len(model.layers) - 1):
        graph.add_edge(f"Layer {i+1}: {model.layers[i]['name']}", f"Layer {i+2}: {model.layers[i+1]['name']}")

    # Draw the graph
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos)

    # Show the plot
    plt.show()


def visualize_performance_metrics(history):
    """
    Visualize the performance metrics.

    Parameters:
    - history: A dictionary containing the training loss and accuracy over epochs.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot the training loss
    ax.plot(history['loss'])
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # Show the plot
    plt.show()


def one_hot_encode(y, num_classes):
    """
    Converts a class vector (integers) to binary class matrix (one-hot encoding).

    Parameters:
    - y: array-like, shape (n_samples,)
        Class labels to be converted.
    - num_classes: int
        Total number of classes.

    Returns:
    - one_hot: array, shape (n_samples, num_classes)
        Binary class matrix.
    """
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Splits arrays or matrices into random train and test subsets.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
        Features to be split.
    - y: array-like, shape (n_samples,)
        Labels to be split.
    - test_size: float, optional (default=0.2)
        Proportion of the dataset to include in the test split.
    - random_state: int, optional (default=None)
        Controls the shuffling applied to the data before applying the split.

    Returns:
    - X_train: array-like, shape (n_train_samples, n_features)
        Training features.
    - X_test: array-like, shape (n_test_samples, n_features)
        Testing features.
    - y_train: array-like, shape (n_train_samples,)
        Training labels.
    - y_test: array-like, shape (n_test_samples,)
        Testing labels.
    """
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def normalize(X):
    """
    Normalizes the dataset to have zero mean and unit variance.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
        Features to be normalized.

    Returns:
    - X_normalized: array-like, shape (n_samples, n_features)
        Normalized features.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


# Example usage:
model = PyTorchModel()
visualize_model_architecture(model)