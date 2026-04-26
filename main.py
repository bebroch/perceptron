

from perceptron import Normalize, Perceptron
import numpy as np

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
