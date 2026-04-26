import numpy as np
from perceptron import Normalize, Perceptron
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

clicked_points = []


def train_perc():
    input_data = []
    targets = []

    for px, py, id in clicked_points:
        input_data.append([px, py])
        targets.append([id])

    normalized_data = normalizer.normalize_train_data(input_data, targets)
    perc.train(normalized_data, batch_size=len(normalized_data), epochs=10)


perceptron_points = []


def construct_curve():
    def delete_last_perceptron_points():
        global perceptron_points
        perceptron_points = []
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(True)
        for px, py, id in clicked_points:
            if id == 1:
                ax.plot(px, py, 'ro')
            elif id == 0:
                ax.plot(px, py, 'bo')

        fig.canvas.draw_idle()

    delete_last_perceptron_points()
    for x in np.arange(0, 11., 0.2):
        for y in np.arange(0, 11., 0.2):
            predict = perc.predict(normalizer.normalize_input([x, y]))
            num = predict[0][0]

            # ax.plot(x, y, 'o', color=(num, num, num))

            if abs(predict - 0.5) <= 0.1:
                ax.plot(x, y, 'ko')


def on_click(event):
    if event.inaxes is None:
        return

    x, y = event.xdata, event.ydata
    id_class = None

    if event.button is MouseButton.LEFT:
        clicked_points.append((x, y, 1))
        print(f"[ЛКМ] Добавлена точка: ({x:.2f}, {y:.2f})")
        ax.plot(x, y, 'ro')
        fig.canvas.draw_idle()
        id_class = 1

    elif event.button is MouseButton.RIGHT:
        clicked_points.append((x, y, 0))
        print(f"[ПКМ] Добавлена точка: ({x:.2f}, {y:.2f})")
        ax.plot(x, y, 'bo')
        fig.canvas.draw_idle()
        id_class = 0

    elif event.button is MouseButton.MIDDLE:
        print(
            f"[СКМ] Проверка точки: ({x:.2f}, {y:.2f}), результат: {perc.predict(normalizer.normalize_input([x, y]))[0]}")

    if not id_class is None:
        train_perc()
    construct_curve()


normalizer = Normalize(input_start=0,
                       input_end=2,
                       output_start=2,
                       output_end=3)

normalizer.set_max_normalize_matrix_2(np.array([10, 10, 1]))


config = {
    "learning_rate": 0.1,
    "layers_config": [
        (2, 3),
        (3, 3),
        (3, 3),
        (3, 1)
    ],
    "train_data_info": {
        "input_data_index": (0, 2),
        "targets_index": (2, 3)
    }
}

perc = Perceptron(config)

# training_data: np._ArrayFloat64_co = normalizer.normalize_train_data(
#     input_data, targets)


fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.grid(True)

fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()

# После закрытия окна печатаем итоговый результат
# print(f"\nИтоговый список точек: {clicked_points}")
