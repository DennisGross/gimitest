import sys
import numpy as np

sys.path.append('../utils/')
from plotting import plot_points


if __name__ == '__main__':
    inputs = np.loadtxt('cp_inputs.txt', delimiter=',')
    bugs = np.loadtxt('cp_bugs.txt', delimiter=',')

    fig, axs = plot_points(bugs[:, [0, 2]])
    fig.savefig('results_mt.png')
    print('----------- Analysis -----------')
    angles = bugs[:, 2]
    first_insight = np.mean(angles[angles > 0.02])
    print('Avg. theta values greater than 0.02:', first_insight)
    tmp = bugs[:, [0, 2]]
    second_insight_data = tmp[(tmp < 0).all(axis=1)]
    second_insight = np.mean(second_insight_data, axis=0)
    print('Negative (x, theta) values:', second_insight)
    print('----------------------')