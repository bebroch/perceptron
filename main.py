
import numpy as np


class Layer:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def get_z(self, x):
        z = np.sum(self.W * x, axis=1, keepdims=True) + self.b
        return z

    def relu_row(self, matrix_row):
        return np.array([max(0, n) for n in matrix_row])

    def relu_matrix(self, matrix):
        return np.array([self.relu_row(row) for row in matrix])

    def fix_error_last_layer(self, learning_rate, delta, a):
        self.W = self.W - learning_rate * delta * a
        self.b = self.b - learning_rate * delta

    def fix_error(self, learning_rate, W_prev, delta_prev, x):
        delta = (W_prev * delta_prev).T
        delta = self.relu_matrix(delta)

        self.W -= learning_rate * delta * x
        self.b -= learning_rate * delta


x = np.array([1, 2, 3])
learning_rate = 0.1
y_true = 0.5

# 1 слой
layer1 = Layer(
    W=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    b=np.array([[0.1], [0.2]])
)

z1 = layer1.get_z(x)
a1 = layer1.relu_matrix(z1).T

# 2 слой
layer2 = Layer(
    W=np.array([[0.7, 0.8]]),
    b=np.array([[0.3]])
)

z2 = layer2.get_z(a1)

# ошибка
# 2 слой
y_pred = z2
delta2 = y_pred - y_true

W2_old = layer2.W
layer2.fix_error_last_layer(learning_rate, delta2, a1)

# 1 слой
layer1.fix_error(learning_rate, W2_old, delta2, x)

print(layer1.W)
print(layer1.b)
print(layer2.W)
print(layer2.b)
