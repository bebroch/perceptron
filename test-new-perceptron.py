import numpy as np
import random as r
import math


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

    @staticmethod
    def get_layer_with_random_weight(
            neurons_input: int,
            neurons_out: int,
            neurons_in_layer: int,
            prev_layer_of_neurons: int,
            learning_rate: float,
            prev_layer: LayerWrapper | None = None):
        limit = np.sqrt(6 / (neurons_input + neurons_out))

        W = np.random.normal(loc=0, scale=limit, size=(
            prev_layer_of_neurons, neurons_in_layer))
        b = np.zeros(shape=(1, neurons_in_layer))

        return LayerWrapper(W=W, b=b, learning_rate=learning_rate, prev_layer=prev_layer)


class Perceptron:
    layers: list[LayerWrapper] = []
    learning_rate: float
    train_data_info: dict

    def __init__(self, config: dict) -> None:
        self.learning_rate = config["learning_rate"]
        self.train_data_info = config["train_data_info"]
        # self.create_layers()
        self._create_rand_layers(config["layers_config"])

    def _create_rand_layers(self, layers_config):
        neurons_input = layers_config[0][0]
        neurons_out = layers_config[-1][1]

        prev_layer = None

        for _, (weight_length, neuron_length) in enumerate(layers_config):
            layer = LayerWrapper.get_layer_with_random_weight(neurons_input=neurons_input,
                                                              neurons_out=neurons_out,
                                                              neurons_in_layer=neuron_length,
                                                              prev_layer_of_neurons=weight_length,
                                                              learning_rate=0.1,
                                                              prev_layer=prev_layer)
            self.layers.append(layer)
            prev_layer = layer

    def create_layers(self):
        W1 = np.array([[0.8, -0.5, 0.3], [-0.2, 0.6, -0.9]])
        W2 = np.array([[0.4], [-0.3], [0.7]])
        b1 = np.array([[0.01, 0.01, 0.01]])
        b2 = np.array([[0.01]])
        layer1 = LayerWrapper(W=W1, b=b1, learning_rate=0.5)
        layer2 = LayerWrapper(
            W=W2, b=b2, learning_rate=0.5, prev_layer=layer1)

        self.layers.append(layer1)
        self.layers.append(layer2)

    def train(self, train_data: np._ArrayFloat64_co, batch_size: int, epochs=1):
        """
        train_data: [
            [[input_data], [targets]],
            ...
        ]
        """
        input_start = self.train_data_info["input_data_index"][0]
        input_end = self.train_data_info["input_data_index"][1]
        target_start = self.train_data_info["targets_index"][0]
        target_end = self.train_data_info["targets_index"][1]

        def get_batch():
            return train_data[0:batch_size]

        def get_input_data(data):
            return np.array([data[input_start:input_end]])

        def get_targets(data):
            return np.array([data[target_start:target_end]])

        perceptron_cost = None
        for epoch_idx in range(epochs):
            np.random.shuffle(train_data)
            batch_data = get_batch()

            for data in batch_data:
                perceptron_cost = self.epoch(
                    get_input_data(data), get_targets(data))

            if (epoch_idx + 1) % max(epochs//10, 1) == 0:
                if perceptron_cost is None:
                    raise Exception("perceptron_cost is None")
                print(
                    f"{epoch_idx + 1}/{epochs} epochs done, cost = {perceptron_cost[0][0]}")

    def epoch(self, input_data, target):
        out = input_data
        for layer in self.layers:
            out = layer.get_a(out)

        self.layers[-1].back_propagation(out - target)
        return (out - target)**2

    def cost_func(self, data):
        pass

    def predict(self, predict_data):
        out = predict_data
        for layer in self.layers:
            out = layer.get_a(out)
        return out


class Normalize:
    max_normalize_matrix: np._ArrayFloat64_co | None = None
    input_start: int
    input_end: int
    output_start: int
    output_end: int

    def __init__(self, input_start, input_end, output_start, output_end) -> None:
        self.input_start = input_start
        self.input_end = input_end
        self.output_start = output_start
        self.output_end = output_end

    def set_max_normalize_matrix_1(self, data):
        self.max_normalize_matrix = data.max(axis=0)

    def set_max_normalize_matrix_2(self, max_matrix):
        self.max_normalize_matrix = max_matrix

    def normalize_data(self, data):
        return data / self.max_normalize_matrix

    def normalize_output(self, out):
        max_matrix = self.max_normalize_matrix
        if max_matrix is None:
            raise Exception("max norm must be set")
        return out * max_matrix[-1]

    def normalize_input(self, input):
        max_matrix = self.max_normalize_matrix
        if max_matrix is None:
            raise Exception("max norm must be set")
        return input / max_matrix[self.input_start:self.input_end]

    def normalize_train_data(self, input_data, targets):
        if len(input_data) != len(targets):
            raise Exception("len(input_data) != len(targets)")

        train_data = np.hstack([input_data, targets], dtype=float)

        if self.max_normalize_matrix is None:
            self.set_max_normalize_matrix_1(train_data)

        return self.normalize_data(train_data)


config = {
    "learning_rate": 0.002,
    "layers_config": [
        (4, 3),   # Входной слой (4 входа -> 3 нейрона)
        (3, 5),
        (5, 1)    # Выходной слой (3 -> 1)
    ],
    "train_data_info": {
        "input_data_index": (0, 4),
        "targets_index": (4, 5)
    }
}


perc = Perceptron(config)
normalizer = Normalize(input_start=0,
                       input_end=4,
                       output_start=4,
                       output_end=5)

input_data = [
    [1, 2, 3, 4],
    [10, 11, 12, 13],
    [5, 6, 7, 8],
    [15, 16, 17, 18],
    [12, 13, 14, 15],
    [25, 26, 27, 28],
    [3, 4, 5, 6],
    [6, 7, 8, 9],
    [8, 9, 10, 11],
]

targets = [
    [5],
    [14],
    [9],
    [19],
    [16],
    [29],
    [7],
    [10],
    [12],
]

# normalizer.set_max_normalize_matrix_2(np.array([10, 100, 100, 100, 100]))

training_data: np._ArrayFloat64_co = normalizer.normalize_train_data(
    input_data, targets)


perc.train(training_data, batch_size=len(training_data), epochs=20000)


input_data = normalizer.normalize_input(np.array([2, 3, 4, 5]))
out = perc.predict(input_data)
print(normalizer.normalize_output(out))
