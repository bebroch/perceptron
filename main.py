from perceptron import Config, LayersConfig, Normalize, Perceptron, TrainData, TrainInfo
import numpy as np

normalizer = Normalize(input_start=0,
                       input_end=4,
                       output_start=4,
                       output_end=5)


config: Config = Config(
    learning_rate=0.1,
    layers_config=LayersConfig(layers_config=[
        (4, 3),
        (3, 3),
        (3, 1)
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

perc.train(training_data,  epochs=7000)
