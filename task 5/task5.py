from dataclasses import dataclass
import json
import json
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # noqa: E402
sys.path.append(os.path.abspath('..'))  # noqa: E402

from matplotlib.widgets import Slider
import numpy as np
from perceptron import Normalize, Perceptron, TrainData
import matplotlib.pyplot as plt
from perceptron import Config, LayersConfig, Normalize, Perceptron, TrainInfo
from data import POINTS_DATA
from numba import njit, prange

perceptron_points = []


def test(config: Config):
    normalizer = Normalize(input_start=0,
                           input_end=2,
                           output_start=2,
                           output_end=3)
    normalizer.set_max_normalize_matrix_2(np.array([2, 2, 1]))

    clicked_points = POINTS_DATA

    def get_normalized_data():
        input_data = []
        targets = []

        for px, py, id in clicked_points:
            input_data.append([px, py])
            targets.append([id])

        return normalizer.normalize_train_data(input_data, targets)

    def train_perc(config: Config):
        normalized_data = get_normalized_data()

        training_data: TrainData = TrainData(
            data=normalized_data,
            batch_size=len(normalized_data)
        )
        perc = Perceptron(config)
        epoch_count = perc.train_to_cost(training_data, cost_threshold=1e-2,
                                         cost_overflow=10, max_epochs=3000)
        return epoch_count

    return train_perc(config)


def approximate_test(config: Config, test_count: int):
    epoches = 0
    for i in range(test_count):
        epoches += test(config)
        print(f"\titeration: {i + 1}, epoch: {epoches}")
    return math.ceil(epoches / test_count)


@njit
def standard_deviation_test(config: Config, test_count: int):
    epoches = np.array([])
    for i in range(test_count):
        np.append(epoches, test(config))
        print(f"\titeration: {i + 1}, epoch: {epoches}")
    return get_standard_deviation(epoches)


def get_standard_deviation(epochs: np.ndarray):
    max_epoch = np.max(epochs)
    return np.sqrt(np.sum((epochs - max_epoch)**2) / len(epochs))


def test_average_on_learning_rate_and_scale(config: Config, learning_rates, scales, test_count=10) -> Plane:

    def add_epoch_to_dependence(epoch_count: int):
        dependence_average_on_learning_rate_and_scale.Z.append(epoch_count)

    dependence_average_on_learning_rate_and_scale: Plane = Plane(
        X=scales,
        Y=learning_rates,
        Z=[]
    )

    for learn_rate in learning_rates:
        for scale in scales:
            print("Testing learning rate:", learn_rate, "scale:", scale)
            config.learning_rate = learn_rate
            config.layers_config.scale = scale
            epoch_count = approximate_test(config, test_count=test_count)
            print(
                f"learning_rate: {learn_rate}, scale: {scale}, epoch: {epoch_count}")
            add_epoch_to_dependence(epoch_count)

    return dependence_average_on_learning_rate_and_scale


def test_RMSE_on_learning_rate_and_scale(config: Config, learning_rates, scales, test_count=10):
    def add_epoch_to_dependence(epoch_count: int):
        dependence_RMSE_on_learning_rate_and_scale.Z.append(epoch_count)

    dependence_RMSE_on_learning_rate_and_scale: Plane = Plane(
        X=scales,
        Y=learning_rates,
        Z=[]
    )

    for learn_rate in learning_rates:
        for scale in scales:
            print("Testing learning rate:", learn_rate, "scale:", scale)
            config.learning_rate = learn_rate
            config.layers_config.scale = scale
            standard_deviation_of_epoch = approximate_test(
                config, test_count=test_count)
            print(
                f"learning_rate: {learn_rate}, scale: {scale}, standard deviation of epoch: {standard_deviation_of_epoch}")
            add_epoch_to_dependence(standard_deviation_of_epoch)

    return dependence_RMSE_on_learning_rate_and_scale


@dataclass
class Plane:
    X: np.ndarray
    Y: np.ndarray
    Z: list[float]

    def get_meshgrid(self):
        return np.meshgrid(self.X, self.Y)

    def get_Z(self):
        return np.array(self.Z).reshape(len(self.Y), len(self.X))


def view_graph(plane: Plane, z_label: str):
    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(111, projection='3d')
    X, Y = plane.get_meshgrid()
    surf = ax.plot_surface(X, Y, plane.get_Z(), cmap='viridis')

    ax_azim = plt.axes((0.2, 0.05, 0.6, 0.03))
    ax_elev = plt.axes((0.2, 0.01, 0.6, 0.03))
    slider_azim = Slider(ax_azim, 'Азимут', 0., 360., valinit=0., valstep=0.1)
    slider_elev = Slider(ax_elev, 'Наклон', -90, 90, valinit=30, valstep=1)

    def update(val):
        ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
        fig.canvas.draw_idle()
    slider_azim.on_changed(update)
    slider_elev.on_changed(update)

    ax.set_xlabel('Scale')
    ax.set_ylabel('Learning Rate')
    ax.set_zlabel(z_label)
    # fig.colorbar(surf)
    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.10, top=1)
    plt.savefig("main.png", bbox_inches='tight')
    plt.show()


def write_to_json(plane: Plane, filename: str):
    with open(filename, "w") as f:
        f.write('{')
        X = json.dumps(plane.X.tolist())
        Y = json.dumps(plane.Y.tolist())
        Z = json.dumps(plane.Z)
        f.write(
            f'"X":{X}, "Y":{Y}, "Z":{Z}')
        f.write('}')


def read_from_json(filename: str):
    with open(filename, "r") as f:
        data = json.load(f)
        X = np.array(data["X"])
        Y = np.array(data["Y"])
        Z = data["Z"]
        return Plane(X=X, Y=Y, Z=Z)


def view_graph_from_json(filename: str, z_label: str):
    plane = read_from_json(filename)
    view_graph(plane, z_label)


def graph_1():
    config: Config = Config(
        learning_rate=0.1,
        layers_config=LayersConfig(
            input_neurons=2,
            hidden_neurons=[3],
            output_neurons=1,
            scale=1.0),
        train_data_info=TrainInfo(
            input_data_index=(0, 2),
            targets_index=(2, 3)
        ),
    )

    learning_rates = np.arange(0.08, 1., .02)
    scales = np.arange(0.1, 3., .02)

    # зависимость среднего количества эпох от learning_rate и scale
    depend = test_average_on_learning_rate_and_scale(
        config, learning_rates, scales, test_count=1)
    write_to_json(depend, "dependence_average_on_learning_rate_and_scale.json")
    view_graph(depend, 'approximate epoch')


def graph_2():
    config: Config = Config(
        learning_rate=0.1,
        layers_config=LayersConfig(
            input_neurons=2,
            hidden_neurons=[3],
            output_neurons=1,
            scale=1.0),
        train_data_info=TrainInfo(
            input_data_index=(0, 2),
            targets_index=(2, 3)
        ),
    )

    learning_rates = np.arange(.08, 1., .05)
    scales = np.arange(.1, 3., .05)

    # зависимость СКО эпох обучения (standard_deviation_of_epoch) от learning_rate и scale
    depend = test_RMSE_on_learning_rate_and_scale(
        config, learning_rates, scales, test_count=1)
    write_to_json(depend, "dependence_RMSE_on_learning_rate_and_scale.json")
    view_graph(depend, 'standard deviation of epoch')


# graph_1()
# view_graph_from_json(
#     'task 5/dependence_average_on_learning_rate_and_scale.json', 'approximate epoch')

# graph_2()
# view_graph_from_json(
#     'task 5/dependence_RMSE_on_learning_rate_and_scale.json', r'$\sqrt{\frac{(epoch - max(epoch))^{2}}{len(epoch)}}$')
