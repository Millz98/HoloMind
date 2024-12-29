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
        return np.dot(self.inputs[0].forward(), self.inputs[1].forward())