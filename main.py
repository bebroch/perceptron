
import numpy as np
import plot
import pickle
from perceptron import Perceptron
from training_data import training_data


def max_number(data):
    flat = np.array(data, dtype=object).flatten()
    all_numbers = []
    for item in flat:
        all_numbers.extend(np.array(item).flatten())
    all_numbers = np.array(all_numbers, dtype=float)
    return all_numbers.max()


max = max_number(training_data)


def normalize_data(data):
    normalized = []
    for item in data:
        # item[0] — список точек [[x,y,label], ...]
        points = np.array(item[0], dtype=float)           # shape (n_points, 3)

        # Нормализуем ТОЛЬКО координаты x и y, label оставляем как есть (0 или 1)
        points[:, :2] = points[:, :2] / max               # делим x и y

        # item[1] — [a, b] для линии (если нужно)
        line_params = np.array(item[1], dtype=float).reshape(-1, 1) / max

        # points теперь (n_points, 3)
        normalized.append([points, line_params])

    return normalized


def save(model: Perceptron, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def load(filename) -> Perceptron:
    with open(filename, "rb") as file:
        return pickle.load(file)


def print_layers(layers):
    for i in range(len(layers)):
        print(i)
        print(layers[i].W)
        print(layers[i].b)


def plot_data(model: Perceptron, dots):
    output_normalized = model.predict(
        np.array(dots).reshape(-1, 1) / max)
    if not output_normalized is None:
        output = output_normalized * max
        print(output[0][0], output[1][0])
        plot.test(dots, output[0][0], output[1][0])
    else:
        print("Error")


def trained_model(epochs):
    model = Perceptron(config)
    model.train(normalize_data(training_data), epochs)
    model.save("my_model.json")
    plot_data(model, dots)


def loaded_model():
    loaded_model = Perceptron.load("my_model.json")
    plot_data(loaded_model, dots)


config = {
    "layers_config": {
        "learning_rate": 0.001,
        "neurons_config": [
            (2, 8),   # Входной слой (3 входа -> 4 нейрона)
            (8, 4),   # Скрытый слой (4 -> 2)
            (4, 1)    # Выходной слой (2 -> 1)
        ]
    }

}

dots = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]

print(max)

trained_model(1000)
# loaded_model()

# model = Perceptron(config)

# model.train(normalize_data(training_data), 20000)

# print(model.layers[0].W)
# plot_data(model, dots)


# save(model, "my_model.pkl")


# loaded_model = load("my_model.pkl")
# print("Тип загруженного объекта:", type(loaded_model))
# print("Количество слоёв:", len(loaded_model.layers))

# print(loaded_model.layers[0].W)
# plot_data(loaded_model, dots)


"""

config = {
    "learning_rate": 0.01,
    "neurons_config": [
        (12, 4),   # Входной слой (3 входа -> 4 нейрона)
        (4, 16),   # Скрытый слой (4 -> 2)
        (16, 4),   # Скрытый слой (4 -> 2)
        (4, 2)    # Выходной слой (2 -> 1)

        # (3, 16),    # Входной слой
        # (16, 16),   # Скрытый слой
        # (16, 16),   # Скрытый слой
        # (16, 4)     # Выходной слой
    ]
}

perc = Perceptron(config)


def loaded_perceptron(dots):
    perc = load("perceptron")
    output_normalized = perc.predict(
        np.array(dots).reshape(-1, 1) / 100)

    if not output_normalized is None:
        output = output_normalized * 100
        print(output[0][0], output[1][0])
        plot.test(dots, output[0][0], output[1][0])


def trained_perceptron(epochs, dots):
    perc.train(normalize_data(training_data), epochs)

    output_normalized = perc.predict(
        np.array(dots).reshape(-1, 1) / 100)
    if not output_normalized is None:
        output = output_normalized * 100
        print(output)
        print(output[0][0], output[1][0])

        plot.test(dots, output[0][0], output[1][0])
        save(perc, "perceptron")


dots = [
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0]
]

trained_perceptron(20000, dots)
# loaded_perceptron(dots)
"""
