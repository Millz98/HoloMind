import numpy as np

class Node:
    def __init__(self, value, parents=None):
        self.value = value  # The value of the node
        self.grad = 0  # The gradient of the node
        self.parents = parents if parents is not None else []  # List of parent nodes
        self.operation = None  # The operation that created this node

    def backward(self):
        # If there are no parents, we are at the root node
        if not self.parents:
            return
        
        # Accumulate gradients for each parent
        for parent in self.parents:
            parent.grad += self.grad * self.operation(parent, self)
            parent.backward()

class Add:
    @staticmethod
    def forward(a, b):
        return Node(a.value + b.value, parents=[a, b])

    @staticmethod
    def backward(a, b):
        return 1, 1  # Gradient of a + b w.r.t a and b

class Multiply:
    @staticmethod
    def forward(a, b):
        return Node(a.value * b.value, parents=[a, b])

    @staticmethod
    def backward(a, b):
        return b.value, a.value  # Gradient of a * b w.r.t a and b

# Example usage
if __name__ == "__main__":
    # Create nodes
    x = Node(2.0)  # x = 2.0
    y = Node(3.0)  # y = 3.0

    # Perform operations
    z = Add.forward(x, y)  # z = x + y
    w = Multiply.forward(z, x)  # w = z * x

    # Set the gradient of the output node (w)
    w.grad = 1.0  # Assume we want to compute the gradient of w w.r.t. x and y

    # Backpropagation
    w.backward()

    # Print gradients
    print(f"Gradient of x: {x.grad}")  # Should print the gradient of w w.r.t x
    print(f"Gradient of y: {y.grad}")  # Should print the gradient of w w.r.t y