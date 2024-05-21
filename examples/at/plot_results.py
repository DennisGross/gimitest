import numpy as np

import sys
sys.path.append('../utils/')
from plotting import plot_fgsm_results


if __name__ == '__main__':
    adversaries = np.loadtxt('adversaries.txt', delimiter=',')
    observations = np.loadtxt('observations.txt', delimiter=',')

    fig, axs = plot_fgsm_results(observations, adversaries)
    fig.savefig('results_fgsm.png')
