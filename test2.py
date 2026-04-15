import numpy as np


def normalize_data(data):
    for i in range(len(data)):
        data[i][0] = np.array(data[i][0]).reshape(-1, 1) / 10
        data[i][1] = np.array(data[i][1]).reshape(-1, 1) / 10
    return data


data = [
    [[1, 2, 3],      [4]],
    [[15, 16, 17],   [18]],
    [[13, 14, 15],   [16]],
    [[5, 6, 7],      [8]],
    [[3, 4, 5],      [6]]
]


print(normalize_data(data))
