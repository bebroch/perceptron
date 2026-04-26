from perceptron import Normalize, Perceptron
import numpy as np
import numpy as np

normalizer = Normalize(input_start=0,
                       input_end=4,
                       output_start=4,
                       output_end=5)


config = {
    "learning_rate": 0.1,
    "layers_config": [
        (4, 3),
        (3, 3),
        (3, 1)
    ],
    "train_data_info": {
        "input_data_index": (0, 4),
        "targets_index": (4, 5)
    }
}

perc = Perceptron(config)

training_data: np._ArrayFloat64_co = normalizer.normalize_train_data(
    input_data, targets)

perc.train(training_data, batch_size=len(training_data), epochs=7000)
