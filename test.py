import numpy as np


def get_z(W, b, x):
    print(W*x)
    z = np.sum(W * x, axis=1, keepdims=True) + b
    return z


x = [1, 2, 3]
# 1 слой
# W1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
# b1 = np.array([[0.1], [0.2]])
# z1 = get_z(W1, b1, x)
# print(np.array_equal(z1, np.array([[1.5], [3.4]])))

# 2 слой
W2 = np.array([[0.7, 0.8]])
b2 = np.array([[0.3]])
a1 = np.array([[1.5, 3.4]])

z2 = get_z(W2, b2, a1)
print(np.array_equal(z2, np.array([[4.0700]])))
print(z2)
