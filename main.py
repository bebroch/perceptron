from perceptron import Config, LayersConfig, Normalize, Perceptron, TrainData, TrainInfo
import numpy as np

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

normalizer = Normalize(input_start=0,
                       input_end=4,
                       output_start=4,
                       output_end=5)


config: Config = Config(
    learning_rate=0.1,
    layers_config=LayersConfig(layers_config=[
        (4, 3),
        (3, 5),
        (5, 5),
        (5, 1)
    ], scale=1.0),
    train_data_info=TrainInfo(input_data_index=(0, 4), targets_index=(4, 5)))


perc = Perceptron(config)


normalized_data = normalizer.normalize_train_data(input_data, targets)
training_data: TrainData = TrainData(
    data=normalized_data,
    batch_size=len(normalized_data)
)


normalizer.normalize_train_data(
    input_data, targets)


perc.train_to_cost(training_data, cost_threshold=1e-3,
                   cost_overflow=10, max_epochs=25000)
