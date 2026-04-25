import numpy as np
import random as r


class Layer:
    W: np._ArrayFloat64_co
    b: np._ArrayFloat64_co

    def __init__(self, W, b):
        self.W = W
        self.b = b

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_z(self, x):
        return x @ self.W + self.b

    def get_a(self, x):
        """а = σ(z) = σ(W*x + b)"""
        return self.sigmoid(self.get_z(x))


class LayerWrapper(Layer):
    prev_layer: LayerWrapper | None
    current_a: np._ArrayFloat64_co | None
    prev_a: np._ArrayFloat64_co | None
    prev_a: np._ArrayFloat64_co | None
    learning_rate: int

    def __init__(self, W, b, learning_rate, prev_layer: LayerWrapper | None = None):
        super().__init__(W, b)
        self.prev_layer = prev_layer
        self.learning_rate = learning_rate
        self.current_a = None

    def get_a(self, x) -> np._ArrayFloat64_co:
        self.prev_a = x
        current_a = super().get_a(x)
        self.current_a = current_a
        return current_a

    def find_error(self, dE_dy_pred):
        current_a = self.current_a
        if current_a is None:
            raise Exception("current_a is must be set")

        dy_pred_dZ = current_a * (1 - current_a)
        dE_dZ = dE_dy_pred * dy_pred_dZ
        return dE_dZ

    def find_deviation(self, dE_dZ):
        prev_a = self.prev_a
        if prev_a is None:
            raise Exception("current_a is must be set")

        dE_dW = prev_a.T @ dE_dZ
        dE_db = dE_dZ
        return (dE_dW, dE_db)

    def fix_error(self, dE_dW, dE_db):
        W = self.W
        b = self.b
        self.W = W - self.learning_rate * dE_dW
        self.b = b - self.learning_rate * dE_db

    def back_propagation(self, dE_dy_pred):
        """dE_dy_pred = y_pred - y_true"""
        old_W = self.W
        dE_dZ = self.find_error(dE_dy_pred)
        (dE_dW, dE_db) = self.find_deviation(dE_dZ)
        self.fix_error(dE_dW, dE_db)

        prev_layer = self.prev_layer
        if not prev_layer is None:
            prev_layer.back_propagation(dE_dZ @ old_W.T)


def normalize_data(data):
    max = data.max(axis=0)
    return data / max


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def layer_handler(W, b, x):
    return sigmoid(x @ W + b)


def cost_func():
    pass


# X = np.array([[50, 30000], [25, 45000], [30, 75000]], dtype=float)
# X = normalize_data(X)


W1 = np.array([[0.8, -0.5, 0.3], [-0.2, 0.6, -0.9]])
W2 = np.array([[0.4], [-0.3], [0.7]])
b1 = np.array([[0.01, 0.01, 0.01]])
b2 = np.array([[0.01]])

learning_rate = 0.5

layer1 = LayerWrapper(W=W1, b=b1, learning_rate=learning_rate)
layer2 = LayerWrapper(
    W=W2, b=b2, learning_rate=learning_rate, prev_layer=layer1)

x = np.array([[0.2, 0.5]])
print(x)

out1 = layer1.get_a(x)
print("out1", out1)
out2 = layer2.get_a(out1)
print("out2", out2)

x1 = layer_handler(W1, b1, x)
print("x1", x1)
x2 = layer_handler(W2, b2, x1)
print("x2", x2)

layer2.back_propagation(out2 - 0.6)
print("W2", layer2.W)
print("b2", layer2.b)

print("W1", layer1.W)
print("b1", layer1.b)
