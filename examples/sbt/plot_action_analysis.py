import sys
import numpy as np

sys.path.append('../utils/')
from plotting import plot_bars


if __name__ == '__main__':

    log_data = [
        np.loadtxt('random_testing_action_distribution.txt'),
        np.loadtxt('es_action_distribution.txt'),
        np.loadtxt('fuzzing_action_distribution.txt')
    ]

    data = np.vstack([100 * (l / sum(l)) for l in log_data])

    color_names = ['deep sky blue', 'fire engine red', 'amber']
    fig, ax = plot_bars(
        data=data,
        labels=['RT', 'ST', 'FT'],
        colors=[f'xkcd:{color}' for color in color_names],
        x_labels=['NOP', 'LEFT', 'MAIN', 'RIGHT']
    )

    fig.savefig('action_distribution.png')
