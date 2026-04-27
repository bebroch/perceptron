from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numba import njit, prange


@dataclass
class TrainInfo:
    input_data_index: tuple[int, int]
    targets_index: tuple[int, int]

    def get_input_start(self):
        return self.input_data_index[0]

    def get_input_end(self):
        return self.input_data_index[1]

    def get_target_start(self):
        return self.targets_index[0]

    def get_target_end(self):
        return self.targets_index[1]


@dataclass
class LayersConfig:
    input_neurons: int
    output_neurons: int
    scale: float = 1.0
    hidden_neurons: list[int] = field(default_factory=list)


@dataclass
class Config:
    learning_rate: float
    layers_config: LayersConfig
    train_data_info: TrainInfo
    is_get_information_about_cost: bool = False


@dataclass
class TrainData:
    data: np._ArrayFloat64_co
    batch_size: int

    def shuffle(self):
        np.random.shuffle(self.data)

    def get_batch(self):
        return self.data[0:self.batch_size]


class Layer:
    W: np._ArrayFloat64_co
    b: np._ArrayFloat64_co

    def __init__(self, W, b):
        self.W = W
        self.b = b

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_z(self, x):
        return fast_get_z(x, self.W, self.b)

    def get_a(self, x):
        """а = σ(z) = σ(W*x + b)"""
        return self.sigmoid(self.get_z(x))


@njit(nogil=True)
def fast_dot(a, b):
    return a.T @ b


@njit(nogil=True)
def fast_get_z(x, W, b):
    return x @ W + b


@njit(nogil=True)
def fast_fix_error(W, b, learning_rate, dE_dW, dE_db):
    W = W - learning_rate * dE_dW
    b = b - learning_rate * dE_db
    return W, b


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

        dE_dW = fast_dot(prev_a, dE_dZ)
        dE_db = dE_dZ
        return (dE_dW, dE_db)

    def fix_error(self, dE_dW, dE_db):
        self.W, self.b = fast_fix_error(
            self.W, self.b, self.learning_rate, dE_dW, dE_db)

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
            scale: float = 1,
            prev_layer: LayerWrapper | None = None):
        limit = np.sqrt(6 / (neurons_input + neurons_out)) * scale

        W = np.random.normal(loc=0, scale=limit, size=(
            prev_layer_of_neurons, neurons_in_layer))
        b = np.zeros(shape=(1, neurons_in_layer))

        return LayerWrapper(W=W, b=b, learning_rate=learning_rate, prev_layer=prev_layer)


class Perceptron:
    def __init__(self, config: Config) -> None:
        self.layers: list[LayerWrapper] = []

        self.is_get_information_about_cost = config.is_get_information_about_cost

        self.learning_rate: float = config.learning_rate
        self.train_info: TrainInfo = config.train_data_info
        self.create_rand_layers(config.layers_config)

    def create_rand_layers(self, layers_config: LayersConfig):
        neurons_input = layers_config.input_neurons
        neurons_out = layers_config.output_neurons

        prev_layer = None

        learning_rate = self.learning_rate
        scale = layers_config.scale

        prev_neuron_count = layers_config.input_neurons
        for neuron_count in layers_config.hidden_neurons + [layers_config.output_neurons]:
            layer = LayerWrapper.get_layer_with_random_weight(neurons_input=neurons_input,
                                                              neurons_out=neurons_out,
                                                              neurons_in_layer=neuron_count,
                                                              prev_layer_of_neurons=prev_neuron_count,
                                                              learning_rate=learning_rate,
                                                              scale=scale,
                                                              prev_layer=prev_layer)
            prev_neuron_count = neuron_count
            self.layers.append(layer)
            prev_layer = layer

    def train(self, train_data: TrainData, epochs=1):
        """
        train_data: [
            [[input_data], [targets]],
            ...
        ]
        """

        input_start = self.train_info.get_input_start()
        input_end = self.train_info.get_input_end()
        target_start = self.train_info.get_target_start()
        target_end = self.train_info.get_target_end()

        def get_input_data(data):
            return np.array([data[input_start:input_end]])

        def get_targets(data):
            return np.array([data[target_start:target_end]])

        def progress_training(epoch_idx):
            if (epoch_idx + 1) % max(epochs//10, 1) != 0:
                return

            if perceptron_cost is None:
                raise Exception("perceptron_cost is None")

            print(
                f"{epoch_idx + 1}/{epochs} epochs done, cost = {perceptron_cost[0][0]}")

        perceptron_cost = None
        for epoch_idx in range(epochs):
            train_data.shuffle()
            batch_data = train_data.get_batch()

            for data in batch_data:
                perceptron_cost = self.epoch(
                    get_input_data(data), get_targets(data))

            progress_training(epoch_idx)

    def train_to_cost(self, train_data: TrainData, cost_threshold=0.01, cost_overflow=10, max_epochs=25000):
        input_start = self.train_info.get_input_start()
        input_end = self.train_info.get_input_end()
        target_start = self.train_info.get_target_start()
        target_end = self.train_info.get_target_end()

        def get_input_data(data):
            return np.array([data[input_start:input_end]])

        def get_targets(data):
            return np.array([data[target_start:target_end]])

        def progress_training(epoch_idx):
            if not self.is_get_information_about_cost:
                return

            if (epoch_idx + 1) % max(max_epochs/5//10, 1) != 0:
                return

            print(
                f"{epoch_idx + 1} epochs done, cost = {perceptron_costs[-1]}")

        def isPerceptronCostMoreThanThreshold():
            for cost in perceptron_costs:
                if cost > cost_threshold:
                    return True
            return False

        def isCostsOverflow():
            return len(perceptron_costs) > cost_overflow

        perceptron_costs = []
        epoch_idx = 0
        while epoch_idx <= max_epochs:
            train_data.shuffle()
            batch_data = train_data.get_batch()

            perc_cost_sum = 0
            for data in batch_data:
                perc_cost = self.epoch(get_input_data(data), get_targets(data))
                perc_cost_sum += perc_cost[0][0]

            perceptron_costs.append(perc_cost_sum / len(batch_data))

            if not isPerceptronCostMoreThanThreshold():
                if self.is_get_information_about_cost:
                    print(
                        f"training done, epochs: {epoch_idx}, cost: {perceptron_costs[-1]}")
                return epoch_idx

            if isCostsOverflow():
                perceptron_costs.pop(0)

            progress_training(epoch_idx)
            epoch_idx += 1

        if self.is_get_information_about_cost:
            print(
                f"training not done, epochs: {epoch_idx}, cost: {perceptron_costs[-1]}")
        return epoch_idx

    def epoch(self, input_data, target):
        out = input_data
        for layer in self.layers:
            out = layer.get_a(out)

        self.layers[-1].back_propagation(out - target)
        return (out - target)**2

    def cost_func(self, data: np._ArrayFloat64_co):
        input_start = self.train_info.get_input_start()
        input_end = self.train_info.get_input_end()
        target_start = self.train_info.get_target_start()
        target_end = self.train_info.get_target_end()

        def get_input_data():
            return np.array([data[input_start:input_end]])

        def get_targets():
            return np.array([data[target_start:target_end]])

        predict = self.predict(get_input_data())
        cost = (predict - get_targets())**2
        return cost[0][0]

    def predict(self, predict_data):
        out = predict_data
        for layer in self.layers:
            out = layer.get_a(out)
        return out


class Normalize:
    max_normalize_matrix: np._ArrayFloat64_co | None
    input_start: int
    input_end: int
    output_start: int
    output_end: int

    def __init__(self, input_start, input_end, output_start, output_end) -> None:
        self.max_normalize_matrix = None

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

    def normalize_target(self, target):
        max_matrix = self.max_normalize_matrix
        if max_matrix is None:
            raise Exception("max norm must be set")
        return target / max_matrix[-1]

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
