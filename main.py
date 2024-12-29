# main.py

import numpy as np
from holomind.models import Model
from holomind.layers import Dense, ReLU, Dropout, BatchNormalization
from holomind.optimizers import SGD, Adam, LearningRateScheduler
from holomind.loss import MeanSquaredError
from holomind.utils import visualize_model_architecture, visualize_performance_metrics

def main():
    # Generate some sample data
    X = np.random.rand(100, 3)  # 100 samples, 3 features
    y = np.random.rand(100, 1)  # 100 target values

    # Create a model
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

    # Compile the model with a loss function and optimizer
    model.compile(loss_function=MeanSquaredError(), optimizer=Adam(learning_rate=0.001))

     # Train the model
    model.fit(X, y, epochs=50)  # Increased epochs

    # Visualize the model architecture
    visualize_model_architecture(model)

    # Visualize the performance metrics
    visualize_performance_metrics(model.history)

    # Print a message indicating training is complete
    print("Training complete!")

if __name__ == "__main__":
    main()