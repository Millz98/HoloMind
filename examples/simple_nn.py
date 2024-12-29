# examples/simple_nn.py
import numpy as np
from holomind.layers import Dense, ReLU, Dropout
from holomind.optimizers import SGD
from holomind.loss import MeanSquaredError
from holomind.models import Model

# Sample data
X = np.random.rand(100, 10)
y = np.random.rand(100, 1)

# Create model
model = Model()
model.add(Dense(2, 2))  # Input layer
model.add(ReLU())        # Activation layer
model.add(Dropout(rate=0.5))  # Dropout layer
model.add(Dense(2, 1))   # Output layer

# Compile model
model.compile(loss_function=MeanSquaredError(), optimizer=SGD(learning_rate=0.01))

# Train model
model.fit(X, y, epochs=100)