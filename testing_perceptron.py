

from perceptron import Normalize, Perceptron
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import random as r
import shutil

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


def test(learning_rate, neuron_count, epoch):
    config = {
        "learning_rate": learning_rate,
        "layers_config": [
            (4, 3),
            (3, 3),
            (3, neuron_count),
            (neuron_count, 1)
        ],
        "train_data_info": {
            "input_data_index": (0, 4),
            "targets_index": (4, 5)
        }
    }

    perc = Perceptron(config)

    training_data: np._ArrayFloat64_co = normalizer.normalize_train_data(
        input_data, targets)

    perc.train(training_data, batch_size=len(training_data), epochs=epoch)
    return perc


def write_to_files(learning_rate, neuron_count, epoch, cost):
    os.makedirs(
        f'./results/learn_{learning_rate}/neuron_{neuron_count}', exist_ok=True)
    with open(f'./results/learn_{learning_rate}/neuron_{neuron_count}/result.txt', 'a', encoding='utf-8') as f:
        f.write(f'{epoch}\t{cost}\n')


def view_graph(learning_rate, neuron_count, epoch, costs):
    def settings():
        ax = plt.gca()
        ax.set_xlabel(r'$epoch$', fontsize=20)
        ax.set_ylabel(r'$cost$', fontsize=20)
        ax.grid(True, which="both", ls="-", color='0.7', alpha=0.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    plt.figure(figsize=(14, 8))

    settings()

    plt.plot(epoch, costs, 'bo-')

    plt.xscale('log')

    plt.savefig(
        f"./results/learn_{learning_rate}/neuron_{neuron_count}/result.png", bbox_inches='tight')


def view_graph_of_learning_rate(learning_rate, neuron_count_list, epoch, costs):
    def settings():
        ax = plt.gca()
        ax.set_xlabel(r'$epoch$', fontsize=20)
        ax.set_ylabel(r'$cost$', fontsize=20)
        ax.grid(True, which="both", ls="-", color='0.7', alpha=0.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    plt.figure(figsize=(14, 8))
    settings()

    for i in range(len(neuron_count_list)):
        color = (r.random(), r.random(), r.random())
        plt.plot(epoch, costs[i], 'ko-', color=color,
                 label=f'neuron count: {neuron_count_list[i]}')

    plt.xscale('log')

    plt.legend(loc="best", fontsize=12, framealpha=1)

    plt.savefig(
        f"./results/learn_{learning_rate}/result.png", bbox_inches='tight')


def main():
    learning_rate_list = [0.005, 0.01, 0.05, 0.1, 0.5]
    neuron_count_list = [1, 2, 3, 5, 10, 20]
    epoch_list = [1, 2, 5, 10, 50, 100, 200, 500, 1000, 5000, 10000, 50000]

    for learning_rate in learning_rate_list:
        learning_rate_costs = []
        for neuron_count in neuron_count_list:
            costs = []
            for epoch in epoch_list:
                perc = test(learning_rate, neuron_count, epoch)
                input_data = normalizer.normalize_input(np.array([2, 3, 4, 5]))
                target = normalizer.normalize_target(np.array([6]))

                string = f"learning_rate: {learning_rate}, neuron_count: {neuron_count}, epoch: {epoch}, cost: {perc.cost_func(input_data, target)}"
                print(string)

                cost = perc.cost_func(input_data, target)
                costs.append(cost)

                write_to_files(learning_rate,
                               neuron_count,
                               epoch, cost
                               )
            view_graph(learning_rate,
                       neuron_count,
                       epoch_list,
                       costs)

            learning_rate_costs.append(costs)

        view_graph_of_learning_rate(
            learning_rate, neuron_count_list, epoch_list, learning_rate_costs)


if __name__ == "__main__":
    main()


# perc = Perceptron(config)


# # normalizer.set_max_normalize_matrix_2(np.array([10, 100, 100, 100, 100]))

# training_data: np._ArrayFloat64_co = normalizer.normalize_train_data(
#     input_data, targets)


# perc.train(training_data, batch_size=len(training_data), epochs=20000)


# input_data = normalizer.normalize_input(np.array([2, 3, 4, 5]))
# out = perc.predict(input_data)
# print(normalizer.normalize_output(out))
