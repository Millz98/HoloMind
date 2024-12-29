import numpy as np

class Operation:
    def __init__(self, *inputs):
        self.inputs = inputs
        self.output = None

    def compute(self):
        """Override this method in subclasses to define the operation."""
        raise NotImplementedError

    def forward(self):
        """Compute the output of the operation."""
        if self.output is None:
            self.output = self.compute()
        return self.output

class MatrixMultiply(Operation):
    def compute(self):
        # Check if inputs are numpy arrays
        input1 = self.inputs[0].forward() if isinstance(self.inputs[0], Operation) else self.inputs[0]
        input2 = self.inputs[1].forward() if isinstance(self.inputs[1], Operation) else self.inputs[1]
        return np.dot(input1, input2)

class Addition(Operation):
    def compute(self):
        input1 = self.inputs[0].forward() if isinstance(self.inputs[0], Operation) else self.inputs[0]
        input2 = self.inputs[1].forward() if isinstance(self.inputs[1], Operation) else self.inputs[1]
        return np.add(input1, input2)

class Subtraction(Operation):
    def compute(self):
        input1 = self.inputs[0].forward() if isinstance(self.inputs[0], Operation) else self.inputs[0]
        input2 = self.inputs[1].forward() if isinstance(self.inputs[1], Operation) else self.inputs[1]
        return np.subtract(input1, input2)

class ElementwiseMultiply(Operation):
    def compute(self):
        input1 = self.inputs[0].forward() if isinstance(self.inputs[0], Operation) else self.inputs[0]
        input2 = self.inputs[1].forward() if isinstance(self.inputs[1], Operation) else self.inputs[1]
        return np.multiply(input1, input2)

class Sigmoid(Operation):
    def compute(self):
        input_data = self.inputs[0].forward() if isinstance(self.inputs[0], Operation) else self.inputs[0]
        return 1 / (1 + np.exp(-input_data))

class Tanh(Operation):
    def compute(self):
        input_data = self.inputs[0].forward() if isinstance(self.inputs[0], Operation) else self.inputs[0]
        return np.tanh(input_data)

# You can add more operations as needed