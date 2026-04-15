import numpy as np
import json


class Layer:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def get_z(self, x):
        z = self.W @ x + self.b
        return z

    def relu_matrix(self, matrix):
        return np.maximum(0, matrix)

    def fix_error_last_layer(self, learning_rate, answer_pred, answer_true, x):
        delta = answer_pred - answer_true
        self.W = self.W - learning_rate * np.dot(delta, x.T)
        self.b = self.b - learning_rate * delta
        return delta

    def fix_error(self, learning_rate, W_next, delta_next, z_current, prev_input):
        delta = np.dot(W_next.T, delta_next)
        relu_derivative = (z_current > 0).astype(np.float32)
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
    layers: list[LayerWrapper] = []
    learning_rate: float

    def __init__(self, config: dict):
        """
        layers_config: конфигурация сети
            learning_rate: скорость обучения
            neurons_config:
                список кортежей (нейроны с предыдущего слоя, количество нейронов в текущем слое)
                Пример: [(3, 4), (4, 2), (2, 1)]
        model_data: данные о слоях сети
            learning_rate: скорость обучения
            layers_data: список кортежей (W, b)
        """

        if "layers_config" in config:
            self.learning_rate = config["layers_config"]["learning_rate"]
            self.create_rand_layers(config["layers_config"]["neurons_config"])
        elif "model_data" in config:
            self.learning_rate = config["model_data"]["learning_rate"]
            self.create_layers(config["model_data"]["layers_data"])

    def randomize_weights(self, output_size, input_size):
        return np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)

    def randomize_biases(self, neurons_count):
        return np.random.randn(neurons_count, 1)

    def create_rand_layers(self, layers_config):
        prev_layer = None

        for _, (input_size, output_size) in enumerate(layers_config):
            layer = LayerWrapper(
                W=self.randomize_weights(output_size, input_size),
                b=self.randomize_biases(output_size),
                prev_layer=prev_layer
            )
            self.layers.append(layer)
            prev_layer = layer

    def create_layers(self, layers_data):
        prev_layer = None

        for layer_data in layers_data:
            layer = LayerWrapper(
                W=layer_data[0],
                b=layer_data[1],
                prev_layer=prev_layer
            )
            self.layers.append(layer)
            prev_layer = layer

    def train(self, data, epochs=1):
        for epoch_idx in range(epochs):
            np.random.shuffle(data)                    # важно! перемешиваем
            for item in data:
                points = item[0]                       # (n_points, 3)
                for point in points:                   # берём по одной точке
                    x = point[:2].reshape(2, 1)        # вход: (2, 1)
                    y_true = point[2].reshape(1, 1)    # метка: (1, 1)
                    self.epoch(x, y_true)

            if (epoch_idx + 1) % max(epochs//10, 1) == 0:
                print(f"{epoch_idx + 1}/{epochs} epochs done")

    def epoch(self,  enter_data, answer_true):
        z = None
        a = enter_data

        for i in range(len(self.layers)):
            self.layers[i].current_input = a
            z = self.layers[i].get_z(a)

            if i == len(self.layers) - 1:
                z_clipped = np.clip(z, -500, 500)
                a = 1 / (1 + np.exp(-z_clipped))
            else:
                a = self.layers[i].relu_matrix(z)

            self.layers[i].z_current = z

        self.layers[-1].fix_error_last_layer_wrapper(
            self.learning_rate, a, answer_true)

    def predict(self, enter_data):
        z = None
        a = enter_data

        for i in range(len(self.layers)):
            z = self.layers[i].get_z(a)
            a = self.layers[i].relu_matrix(z)

        return z

    def save(self, path):
        obj = []
        for layer in self.layers:
            obj.append((layer.W.tolist(), layer.b.tolist()))
        with open(path, "w") as file:
            file.write(json.dumps(
                {"learning_rate": self.learning_rate, "layers_data": obj}))

    @staticmethod
    def load(path):
        loaded_model = None
        with open(path, "r") as file:
            loaded_model = json.loads(file.read())

        model_data = {
            "learning_rate": loaded_model["learning_rate"],
            "layers_data": []
        }
        for layer in loaded_model["layers_data"]:
            model_data["layers_data"].append(
                (np.array(layer[0]), np.array(layer[1])))

        return Perceptron({"model_data": model_data})
