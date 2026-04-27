import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # noqa: E402
sys.path.append(os.path.abspath('..'))  # noqa: E402

import numpy as np
from perceptron import Normalize, Perceptron, TrainData
import matplotlib.pyplot as plt
from perceptron import Config, LayersConfig, Normalize, Perceptron,  TrainInfo
from data import POINTS_DATA

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
        epoch_count = perc.train_to_cost(training_data, cost_threshold=5e-3,
                                         cost_overflow=10, max_epochs=100000)
        return epoch_count

    return train_perc(config)


def approximate_test(config: Config, test_count: int = 10):
    epoches = 0
    for i in range(test_count):
        epoches += test(config)
        print(f"\titeration: {i+1}, epoch: {epoches}")
    return epoches / test_count


def test_learning_rate(config: Config):
    learning_rates = np.arange(0.08, 1., 0.01)

    def add_epoch_to_dependence(learning_rate, epoch_count):
        dependence_epoch_on_learning_rate.append((epoch_count, learning_rate))

    dependence_epoch_on_learning_rate: list[tuple[int, float]] = []

    for learn_rate in learning_rates:
        print("Testing learning rate:", learn_rate)
        config.learning_rate = learn_rate
        epoch_count = approximate_test(config)
        print(f"learning_rate: {learn_rate}, epoch: {epoch_count}")
        add_epoch_to_dependence(learn_rate, epoch_count)

    print("Dependence epoch on learning rate:")
    for epoch, learn_rate in dependence_epoch_on_learning_rate:
        print(f"learning_rate: {learn_rate}, epoch: {epoch}")

    return dependence_epoch_on_learning_rate


def test_scale(config: Config):
    scales = np.arange(0.1, 3., 0.05)

    def add_epoch_to_dependence(scale, epoch_count):
        dependence_epoch_on_scale.append((epoch_count, scale))

    dependence_epoch_on_scale: list[tuple[int, float]] = []

    for scale in scales:
        print("Testing scale:", scale)
        config.layers_config.scale = scale
        epoch_count = approximate_test(config)
        print(f"scale: {scale}, epoch: {epoch_count}")
        add_epoch_to_dependence(scale, epoch_count)

    print("Dependence epoch on scale:")
    for epoch, scale in dependence_epoch_on_scale:
        print(f"scale: {scale}, epoch: {epoch}")

    return dependence_epoch_on_scale


def view_graph(dependences, x_label):
    plt.figure(figsize=(16, 9))
    epochs, learning_rates = zip(*dependences)
    plt.plot(learning_rates, epochs, 'ro-',
             label=f"Neurons in hidden layer: 3")

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('Epochs', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.legend(loc="best", fontsize=16)
    plt.savefig("main.png", bbox_inches='tight')
    plt.show()


def main():

    config: Config = Config(
        learning_rate=0.1,
        layers_config=LayersConfig(
            layers_config=[
                (2, 3),
                (3, 1)
            ],
            scale=1.0),
        train_data_info=TrainInfo(
            input_data_index=(0, 2),
            targets_index=(2, 3)
        )
    )

    # зависимость количества эпох от learning_rate
    depend = test_learning_rate(config)
    view_graph(depend, 'Learning Rate')

    # зависимость количества эпох от learning_rate
    # depend = test_scale(config)
    # view_graph(depend, 'Scale')

    # points = [
    #     (1, 1, 0),
    #     (1, 2, 0),
    #     (2, 1, 0),
    #     (2, 2, 1),
    # ]

    # for x, y, id_class in points:
    #     style = None
    #     if id_class == 1:
    #         style = 'ro'
    #     else:
    #         style = 'bo'

    #     ax.plot(x, y, style, markersize=15)
    #     clicked_points.append((x, y, id_class))

    # train_perc()
    # construct_curve()


main()


plt.show()
