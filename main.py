
import numpy as np


class Layer:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def get_z(self, x):
        z = self.W @ x + self.b
        return z

    def relu_row(self, matrix_row):
        return np.array([max(0, n) for n in matrix_row])

    def relu_matrix(self, matrix):
        return np.array([self.relu_row(row) for row in matrix])

    def fix_error_last_layer(self, learning_rate, answer_pred, answer_true, x):
        delta = answer_pred - answer_true
        self.W = self.W - learning_rate * np.dot(delta, x.T)
        self.b = self.b - learning_rate * delta
        return delta

    def fix_error(self, learning_rate, W_next, delta_next, z_current, prev_input):
        delta = np.dot(W_next.T, delta_next)
        relu_derivative = (z_current > 0).astype(float)
        delta = delta * relu_derivative
        self.W -= learning_rate * np.dot(delta, prev_input.T)
        self.b -= learning_rate * delta
        return delta


class LayerWrapper(Layer):
    current_input = None
    z_current = None

    def __init__(self, W, b, prev_layer: LayerWrapper | None = None):
        super().__init__(W, b)
        self.prev_layer: LayerWrapper | None = prev_layer

    def fix_error_last_layer_wrapper(self, learning_rate, answer_pred, answer_true):
        delta = super().fix_error_last_layer(
            learning_rate, answer_pred, answer_true, self.current_input)

        if not self.prev_layer:
            return

        self.prev_layer.fix_error_wrapper(
            learning_rate, self.W, delta)

    def fix_error_wrapper(self, learning_rate, W_next, delta_next):
        delta = super().fix_error(learning_rate, W_next,
                                  delta_next, self.z_current, self.current_input)

        if not self.prev_layer:
            return

        self.prev_layer.fix_error_wrapper(
            learning_rate, self.W, delta)


class Perceptron:
    learning_rate = 0.01
    layers: list[LayerWrapper] = []

    def __init__(self, layers_config):
        """
        layers_config: список кортежей (нейроны с предыдущего слоя, количество нейронов в текущем слое)
        Пример: [(3, 4), (4, 2), (2, 1)]
        """
        self.create_layers(layers_config)

    def randomize_weights(self, neurons_count, biases_count):
        return np.random.randn(neurons_count, biases_count)

    def randomize_biases(self, neurons_count):
        return np.random.randn(neurons_count, 1)

    def create_layers(self, layers_config):
        prev_layer = None

        for _, (input_size, output_size) in enumerate(layers_config):
            layer = LayerWrapper(
                W=self.randomize_weights(output_size, input_size),
                b=self.randomize_biases(output_size),
                prev_layer=prev_layer
            )
            self.layers.append(layer)
            prev_layer = layer

    def train(self, data, epochs=1):
        for i in range(epochs):
            for [question, answer] in data:
                self.epoch(question, answer)
            if i*100 % epochs == 0:
                print(int(i/epochs*100)+1, "%")

    def predict(self, enter_data):
        z = None
        a = enter_data

        for i in range(len(self.layers)):
            z = self.layers[i].get_z(a)
            a = self.layers[i].relu_matrix(z)

        return z

    def epoch(self,  enter_data, answer_true):
        z_array = []
        a_array = [enter_data]

        for i in range(len(self.layers)):
            self.layers[i].current_input = a_array[i]
            z = self.layers[i].get_z(a_array[i])
            a = self.layers[i].relu_matrix(z)
            self.layers[i].z_current = z

            z_array.append(z)
            a_array.append(a)

        self.layers[-1].fix_error_last_layer_wrapper(
            self.learning_rate, z_array[-1], answer_true)


config = [
    (3, 4),   # Входной слой (3 входа -> 4 нейрона)
    (4, 4),   # Скрытый слой (4 -> 2)
    (4, 4),   # Скрытый слой (4 -> 2)
    (4, 4),   # Скрытый слой (4 -> 2)
    (4, 4)    # Выходной слой (2 -> 1)
]

perc = Perceptron(config)


def normalize_data(data):
    for i in range(len(data)):
        data[i][0] = np.array(data[i][0]).reshape(-1, 1) / 100
        data[i][1] = np.array(data[i][1]).reshape(-1, 1) / 100
    return data


data = [
    [[1, 2, 3],      [4, 5, 6, 7]],
    [[15, 16, 17],   [18, 19, 20, 21]],
    [[13, 14, 15],   [16, 17, 18, 19]],
    [[5, 6, 7],      [8, 9, 10, 11]],
    [[3, 4, 5],      [6, 7, 8, 9]],
    [[0, 1, 2],      [3, 4, 5, 6]],
    [[1, 2, 3],      [4, 5, 6, 7]],
    [[2, 3, 4],      [5, 6, 7, 8]]
]


perc.train(normalize_data(data), 30000)


output = perc.predict(np.array([2, 3, 4]).reshape(-1, 1) / 100)
print(output * 100)
