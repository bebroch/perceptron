import numpy as np
import unittest


def relu_row(matrix_row):
    return np.array([max(0, n) for n in matrix_row])


def relu_matrix(matrix):
    return np.array([relu_row(row) for row in matrix])


class Layer:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def get_z(self, x):
        z = self.W @ x + self.b.T
        # z = np.sum(self.W * x, axis=1, keepdims=True) + self.b
        return z.T

    def get_a(self, z):
        a_pre = []
        for i in range(len(z)):
            a_pre.append(max(0, z[i]))
        a = np.array(a_pre).T
        return a

    def fix_error_last_layer(self, learning_rate, delta, a):
        self.W = self.W - learning_rate * delta * a
        self.b = self.b - learning_rate * delta

    def fix_error(self, learning_rate, W_prev, delta_prev, a):
        delta = (W_prev * delta_prev).T
        delta = relu_matrix(delta)

        self.W -= learning_rate * delta * a
        self.b -= learning_rate * delta


class TestMath(unittest.TestCase):
    def test_get_z(self):
        W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        b = np.array([[0.1], [0.2]])
        layer = Layer(W, b)
        z = layer.get_z(np.array([1, 2, 3]))
        equal = np.array_equal(z, np.array([[1.5], [3.4]]))
        self.assertTrue(equal)

    def test_get_a(self):
        W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        b = np.array([[0.1], [0.2]])
        layer = Layer(W, b)
        z = layer.get_z([1, 2, 3])
        a = layer.get_a(z)
        equal = np.array_equal(a, np.array([[1.5, 3.4]]))
        self.assertTrue(equal)

    def test_fix_error(self):
        x = np.array([1, 2, 3])
        W1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        W2 = np.array([[0.7, 0.8]])
        b = np.array([[0.1], [0.2]])
        learning_rate = 0.1
        layer = Layer(W1, b)
        delta_prev = np.array([[3.57]])
        layer.fix_error(learning_rate, W2, delta_prev, x)

        equal_W = np.allclose(layer.W, np.array(
            [[-0.1499, -0.2998, -0.4497],
             [0.1144, -0.0712, -0.2568]]
        ))
        equal_b = np.allclose(layer.b, np.array(
            [[-0.1499],
             [-0.0856]]
        ))
        self.assertTrue(equal_W)
        self.assertTrue(equal_b)


# W1_new[[-0.1499 - 0.2998 - 0.4497]
#        [0.1144 - 0.0712 - 0.2568]]
# b1_new[[-0.1499]
#        [-0.0856]]
# delta1[[2.499]
#        [2.856]]
# delta1 = array([[2.499],
#                 [2.856]])

if __name__ == '__main__':
    unittest.main()
