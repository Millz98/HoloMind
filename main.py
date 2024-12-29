# main.py

import numpy as np
from holomind.models import Model
from holomind.layers import Dense, ReLU, Dropout
from holomind.optimizers import SGD
from holomind.loss import MeanSquaredError

def main():
    # Generate some sample data
    X = np.random.rand(100, 3)  # 100 samples, 3 features
    y = np.random.rand(100, 1)  # 100 target values

    # Create a model
    model = Model()

    # Add layers to the model
    model.add(Dense(input_size=3, output_size=5))  # Input layer
    model.add(ReLU())                               # Activation layer
    model.add(Dropout(rate=0.2))                   # Dropout layer
    model.add(Dense(input_size=5, output_size=1))  # Output layer

    # Compile the model with a loss function and optimizer
    model.compile(loss_function=MeanSquaredError(), optimizer=SGD(learning_rate=0.01))

    # Train the model
    model.fit(X, y, epochs=10)

    # Print a message indicating training is complete
    print("Training complete!")

if __name__ == "__main__":
    main()
