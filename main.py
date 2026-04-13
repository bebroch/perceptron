
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

    def fix_error_last_layer(self, learning_rate, answer_pred, answer_true, x):
        delta = answer_pred - answer_true
        self.W = self.W - learning_rate * delta * x
        self.b = self.b - learning_rate * delta
        return delta

    def fix_error(self, learning_rate, W_prev, delta_prev, x):
        delta = (W_prev * delta_prev).T
        delta = self.relu_matrix(delta)

        self.W -= learning_rate * delta * x
        self.b -= learning_rate * delta

        return delta


class LayerWrapper(Layer):
    def __init__(self, W, b, prev_layer: LayerWrapper | None = None):
        super().__init__(W, b)
        self.prev_layer: LayerWrapper | None = prev_layer

    def fix_error_last_layer_wrapper(self, learning_rate, answer_pred, answer_true, all_x):
        delta = super().fix_error_last_layer(
            learning_rate, answer_pred, answer_true, all_x[-1])
        if not self.prev_layer:
            return

        self.prev_layer.fix_error_wrapper(
            learning_rate, self.W, delta, all_x[0:-1])

    def fix_error_wrapper(self, learning_rate, W_prev, delta_prev, all_x):
        delta = super().fix_error(learning_rate, W_prev, delta_prev, all_x[-1])
        if not self.prev_layer:
            return

        self.prev_layer.fix_error_wrapper(
            learning_rate, self.W, delta, all_x[0:-1])


class Perceptron:
    learning_rate = 0.1
    enter_layer: LayerWrapper
    hidden_layers: list[LayerWrapper] = []
    output_layer: LayerWrapper

    def __init__(self, enter_data_count, enter_neurons_count, hidden_layers_count, output_neurons_count):
        self.create_layers(enter_data_count, enter_neurons_count,
                           hidden_layers_count, output_neurons_count)

    def randomize_weights(self, neurons_count, biases_count):
        return np.random.rand(neurons_count, biases_count)

    def randomize_biases(self, neurons_count):
        return np.random.rand(neurons_count, 1)

    def create_layers(self, enter_data_count, enter_neurons_count, hidden_layers_count, output_neurons_count):
        self.enter_layer = LayerWrapper(
            W=self.randomize_weights(
                enter_data_count, enter_neurons_count),
            b=self.randomize_biases(enter_neurons_count)
        )

        prev_layer = self.enter_layer

        for i in range(hidden_layers_count):
            prev_layer = LayerWrapper(
                W=self.randomize_weights(
                    enter_data_count, enter_neurons_count),
                b=self.randomize_biases(enter_neurons_count),
                prev_layer=prev_layer
            )

            self.hidden_layers.append(prev_layer)

        self.output_layer = LayerWrapper(
            W=self.randomize_weights(
                output_neurons_count, 10),
            b=self.randomize_biases(output_neurons_count),
            prev_layer=prev_layer
        )

    def train(self, enter_data, answer_true):
        all_layers = [self.enter_layer, *self.hidden_layers, self.output_layer]
        for i in range(100):
            self.epoch(all_layers, enter_data, answer_true)

    def epoch(self, all_layers, enter_data, answer_true):
        z = None
        a = enter_data

        z_array = []
        a_array = [enter_data]

        for i in range(len(all_layers)):
            z = all_layers[i].get_z(a_array[i])
            a = all_layers[i].relu_matrix(z)

            z_array.append(z)
            a_array.append(a)

        all_layers[-1].fix_error_last_layer_wrapper(
            self.learning_rate, z_array[-1], answer_true, a_array[0:-1])


perc = Perceptron(3, 3, 1, 1)
enter_data = np.array([1, 2, 3])
answer_true = np.array([4])
perc.train(enter_data, answer_true)
