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


class Perceptron:
    layers: list[LayerWrapper] = []
    learning_rate: float
    train_data_info: dict

    def __init__(self, config: dict) -> None:
        self.learning_rate = config["learning_rate"]
        self.train_data_info = config["train_data_info"]
        self.create_rand_layers(config["layers_config"])

    def create_rand_layers(self, layers_config):
        neurons_input = layers_config[0][0]
        neurons_out = layers_config[-1][1]

        prev_layer = None

        for _, (weight_length, neuron_length) in enumerate(layers_config):
            layer = get_layer_with_random_weight(neurons_input=neurons_input,
                                                 neurons_out=neurons_out,
                                                 neurons_in_layer=neuron_length,
                                                 prev_layer_of_neurons=weight_length,
                                                 learning_rate=0.1,
                                                 prev_layer=prev_layer)
            self.layers.append(layer)
            prev_layer = layer

    def train(self, train_data: list, epochs=1):
        """
        train_data: [
            [[input_data], [targets]],
            ...
        ]
        """

        def get_batch():
            return train_data[0:math.ceil(np.size(train_data, 0)*0.2)]

        input_start = self.train_data_info["input_data_index"][0]
        input_end = self.train_data_info["input_data_index"][1]
        target_start = self.train_data_info["targets_index"][0]
        target_end = self.train_data_info["targets_index"][1]

        perceptron_cost = None
        for epoch_idx in range(epochs):
            np.random.shuffle(train_data)
            batch_data = train_data[0:3]

            for data in batch_data:
                perceptron_cost = self.epoch(data[0], data[1])

            if (epoch_idx + 1) % max(epochs//10, 1) == 0:
                print(
                    f"{epoch_idx + 1}/{epochs} epochs done, cost = {perceptron_cost}")

    def epoch(self, input_data, target):
        out = input_data
        for layer in self.layers:
            out = layer.get_a(np.array([out]))

        self.layers[-1].back_propagation(out - target)
        return (out - target)**2

    def cost_func(self, data):
        pass

    def predict(self, predict_data):
        out = predict_data
        for layer in self.layers:
            out = layer.get_a(out)
        return out


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


max_normalize_matrix = None


def normalize_data(data):
    global max_normalize_matrix
    max_normalize_matrix = data.max(axis=0)
    return data / max_normalize_matrix


def normalize_train_data(train_data):
    input_data = np.empty((0, len(train_data[0][0])), dtype=float)
    targets = np.empty((0, len(train_data[0][1])), dtype=float)

    for row in train_data:
        input_data = np.vstack([input_data, row[0]])

    for row in train_data:
        targets = np.vstack([targets, row[1]])

    train_data = np.hstack([input_data, targets])

    return normalize_data(train_data)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def layer_handler(W, b, x):
    return sigmoid(x @ W + b)


def cost_func():
    pass


layer = get_layer_with_random_weight(neurons_input=2,
                                     neurons_out=1,
                                     neurons_in_layer=3,
                                     prev_layer_of_neurons=2,
                                     learning_rate=0.1)


config = {
    "learning_rate": 0.5,
    "layers_config": [
        (4, 3),   # Входной слой (4 входа -> 3 нейрона)
        (3, 1)    # Выходной слой (3 -> 1)
    ],
    "train_data_info": {
        "input_data_index": (0, 4),
        "targets_index": (4, 5)
    }
}


perc = Perceptron(config)

X = [
    [[1, 2, 3, 4], [5]],
    [[10, 11, 12, 13], [14]],
    [[5, 6, 7, 8], [9]],
    [[15, 16, 17, 18], [19]],
    [[12, 13, 14, 15], [16]],
    [[25, 26, 27, 28], [29]],
    [[3, 4, 5, 6], [7]],
]

training_data = [
    [np.array([0.04,   0.07692308, 0.11111111, 0.14285714]),
     np.array([0.17241379])],
    [np.array([0.4,  0.42307692, 0.44444444, 0.46428571]),
     np.array([0.48275862])],
    [np.array([0.2,  0.23076923, 0.25925926, 0.28571429]),
     np.array([0.31034483])],
    [np.array([0.6,  0.61538462, 0.62962963, 0.64285714]),
     np.array([0.65517241])],
    [np.array([0.48,  0.5, 0.51851852, 0.53571429]), np.array([0.55172414])],
    [np.array([1., 1., 1., 1.]), np.array([1.])],
    [np.array([0.12,   0.15384615, 0.18518519, 0.21428571]),
     np.array([0.24137931])]
]
# normalize_train_data(X)


perc.train(training_data, epochs=100)

out = perc.predict(np.array([2, 3, 4, 5]))
print(out)

# X = np.array([[50, 30000], [25, 45000], [30, 75000]], dtype=float)
# X = normalize_data(X)

# W1 = np.array([[0.8, -0.5, 0.3], [-0.2, 0.6, -0.9]])
# W2 = np.array([[0.4], [-0.3], [0.7]])
# b1 = np.array([[0.01, 0.01, 0.01]])
# b2 = np.array([[0.01]])

# learning_rate = 0.5

# layer1 = LayerWrapper(W=W1, b=b1, learning_rate=learning_rate)
# layer2 = LayerWrapper(
#     W=W2, b=b2, learning_rate=learning_rate, prev_layer=layer1)

# x = np.array([[0.2, 0.5]])
# print(x)

# out1 = layer1.get_a(x)
# print("out1", out1)
# out2 = layer2.get_a(out1)
# print("out2", out2)

# x1 = layer_handler(W1, b1, x)
# print("x1", x1)
# x2 = layer_handler(W2, b2, x1)
# print("x2", x2)

# layer2.back_propagation(out2 - 0.6)
# print("W2", layer2.W)
# print("b2", layer2.b)

# print("W1", layer1.W)
# print("b1", layer1.b)
