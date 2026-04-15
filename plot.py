import matplotlib.pyplot as plt
import numpy as np


def test(dots, a, b):

    x = np.linspace(-1, 2, 1000)
    plt.plot(x, a*x+b, 'ro')

    for dot in dots:
        if dot[2] == 1:
            plt.plot([dot[0]], [dot[1]], 'ro')
        else:
            plt.plot([dot[0]], [dot[1]], 'bo')

    # plt.ylim(-1, 2)
    # plt.xlim(-1, 2)

    plt.show()
