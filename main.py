
import numpy as np


def get_z(W, b, x):
    z = np.sum(W * x, axis=1, keepdims=True) + b
    return z


def get_a(z):
    a_pre = []
    for i in range(len(z)):
        a_pre.append(max(0, z[i]))
    a = np.array(a_pre).T
    return a


x = [1, 2, 3]
learning_rate = 0.1
y_true = 0.5

# 1 слой
W1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
b1 = np.array([[0.1], [0.2]])

z1 = get_z(W1, b1, x)
a1 = get_a(z1)

# 2 слой
W2 = np.array([[0.7, 0.8]])
b2 = np.array([[0.3]])

z2 = get_z(W2, b2, a1)
print("z2", z2)

# ошибка
# 2 слой
y_pred = z2
delta2 = y_pred - y_true

W2_new = W2 - learning_rate * delta2 * a1
b2_new = b2 - learning_rate * delta2

print("W2_new", W2_new)
print("b2_new", b2_new)

# 1 слой
delta1 = (W2 * delta2).T
delta1 = np.array([max(0, delta1[0]), max(0, delta1[1])])
print("delta1", delta1)

W1_new = W1 - learning_rate * delta1 * x
b1_new = b1 - learning_rate * delta1
print("W1_new", W1_new)
print("b1_new", b1_new)
