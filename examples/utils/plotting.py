import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple, Union
from PIL import Image


NUM_TICKS = 5
POINT_SIZE = 5

FAILURE_COLOR = 'xkcd:fire engine red'
PASS_COLOR = 'xkcd:shamrock green'
BLUE_COLOR = 'xkcd:deep sky blue'

ITER_LABEL = '#Iterations'
FAILURE_LABEL = '#Failures'

# X_LABEL = 'Horizontal position of the car'
# Y_LABEL = 'Velocity of the car'
X_LABEL = '$x_{car}$'
Y_LABEL = '$\dot x_{car}$'

MARKER_SIZE_LEGEND = 75.0
LABEL_FONTSIZE = 15

def plot_inputs(inputs, rewards, failures):
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    # plots the points
    ax = axs[0]
    ax.scatter(inputs[:, 0], inputs[:, 1], color='xkcd:deep sky blue', s=POINT_SIZE)
    ax = axs[1]
    im = ax.scatter(inputs[:, 0], inputs[:, 1], c=rewards, s=POINT_SIZE)
    fig.colorbar(im, ax=ax, orientation='vertical', label='Rewards')
    ax = axs[2]

    ax.scatter(inputs[failures, 0], inputs[failures, 1], color=FAILURE_COLOR, s=POINT_SIZE, label='Failures')
    ax.scatter(inputs[~failures, 0], inputs[~failures, 1], color=PASS_COLOR, s=POINT_SIZE, label='Pass')
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncols=2, framealpha=1.0,
        prop={'size': 13}
        # labelspacing=1.0, handletextpad=1.0, borderpad=0.55, borderaxespad=0.7)
    )
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor('lightgray')
    legend_frame.set_edgecolor('black')
    for handle in legend.legend_handles:
        handle.set_sizes([MARKER_SIZE_LEGEND])

    fig.tight_layout()
    return fig, axs


def plot_faults(inputs, rewards, failures):
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    # plots the accumulation of failures
    ax = axs[0]
    failure_acc = count_failures(failures)
    ax.plot(
        np.arange(len(failure_acc)),
        failure_acc,
        color=FAILURE_COLOR
    )
    ax.set_xlabel(ITER_LABEL, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(FAILURE_LABEL, fontsize=LABEL_FONTSIZE)

    ax = axs[1]
    im = ax.scatter(inputs[:, 0], inputs[:, 1], c=rewards, s=POINT_SIZE)
    fig.colorbar(im, ax=ax, orientation='vertical', label='Rewards')

    ax = axs[2]
    ax.scatter(inputs[~failures, 0], inputs[~failures, 1], color=PASS_COLOR, s=POINT_SIZE, label='Pass')
    ax.scatter(inputs[failures, 0], inputs[failures, 1], color=FAILURE_COLOR, s=POINT_SIZE, label='Failures')
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncols=2,
        framealpha=1.0,
        prop={'size': 13}
        # labelspacing=1.0, handletextpad=1.0, borderpad=0.55, borderaxespad=0.7)
    )
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor('lightgray')
    legend_frame.set_edgecolor('black')
    for handle in legend.legend_handles:
        handle.set_sizes([MARKER_SIZE_LEGEND])
    fig.tight_layout()
    return fig, axs


def plot_inputs_in_a_single_chart(inputs):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(inputs[:, 0], inputs[:, 1], color=PASS_COLOR, s=POINT_SIZE, label='Pass')
    return fig, ax


def add_inputs_to_chart(ax, inputs):
    ax.scatter(inputs[:, 0], inputs[:, 1], color=PASS_COLOR, s=POINT_SIZE)
    return ax


def add_faults_to_chart(ax, inputs, failures, label, color):
    ax.scatter(inputs[failures, 0], inputs[failures, 1], color=color, s=POINT_SIZE, label=label)
    return ax


def post_process_figure(fig, ax):
    ax.set_xlabel('$F_{x}$', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('$F_{y}$', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=13)
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncols=4,
        framealpha=1.0,
        prop={'size': 13},
        columnspacing=0.2,
        handletextpad=0.2
        )
    #, labelspacing=1.0, handletextpad=1.0, borderpad=0.55, borderaxespad=0.7)
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor('lightgray')
    legend_frame.set_edgecolor('black')
    for handle in legend.legend_handles:
        handle.set_sizes([MARKER_SIZE_LEGEND])
    fig.tight_layout()
    return fig, ax


def count_failures(oracles: np.ndarray):
    failure_accumulator = []
    num_failures = 0
    for o in oracles:
        num_failures += int(o)
        failure_accumulator.append(num_failures)
    return np.array(failure_accumulator)


def plot_fgsm_results(observations: np.ndarray, adversaries: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(observations[:, 0], observations[:, 1], s=5, color=PASS_COLOR, label='Pass')
    ax.scatter(adversaries[:, 0], adversaries[:, 1], s=5, color=FAILURE_COLOR, label='Adversaries')

    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncols=2,
        framealpha=1.0
    )
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor('lightgray')
    legend_frame.set_edgecolor('black')
    for handle in legend.legend_handles:
        handle.set_sizes([MARKER_SIZE_LEGEND])

    ax.set_xlim(-1.0, 0.6)
    ax.set_xlabel(X_LABEL, fontsize=LABEL_FONTSIZE + 3)
    ax.set_ylim(-0.05, 0.05)
    ax.set_ylabel(Y_LABEL, fontsize=LABEL_FONTSIZE + 3)
    ax.set_xticks(np.linspace(-1.0, 0.6, num=NUM_TICKS))
    ax.set_yticks(np.linspace(-0.05, 0.05, num=NUM_TICKS))
    ax.tick_params(axis='both', labelsize=13)
    # ax.locator_params(axis='both', nbins=4)
    fig.tight_layout()
    return fig, ax


def save_gif(images: List[np.ndarray], filepath: str, duration: float = 8.0):
    if len(images) == 0:
        return
    imgs = [Image.fromarray(img) for img in images]
    imgs[0].save(filepath, save_all=True, append_images=imgs[1:], duration=duration)



def plot_points(data: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 6))
    for (x, y) in data:
        ax.scatter(x, y, color=BLUE_COLOR, s=POINT_SIZE)
    ax.set_xlabel('Initial position of the cart', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Initial angle of the pole', fontsize=LABEL_FONTSIZE)
    # ax.set_xlabel('$x_{initial}$', fontsize=LABEL_FONTSIZE)
    # ax.set_ylabel('$\\theta_{initial}$', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(axis='both', color='0.9', linestyle='-', linewidth=1)
    fig.tight_layout()
    return fig, ax


def plot_lines(data: np.ndarray, length: float):
    fig, ax = plt.subplots(figsize=(6, 6))
    for (x, theta) in data:
        ax.plot([x, x + length * np.sin(theta)], [0.0, length * np.cos(theta)], linewidth=1, c='black', alpha=0.8)
        ax.scatter(x, 0.0, color=BLUE_COLOR, s=POINT_SIZE)

    ax.set_xlabel('Initial position of the cart', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Initial angle of the pole', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=13)
    fig.tight_layout()
    return fig, ax


def plot_bars(
    data: List[np.ndarray],
    labels: List[str],
    colors: List[Union[str, Tuple[float, float, float, float]]],
    x_labels: Optional[List[str]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:

    num_data = len(data)
    bar_width = 0.3
    x = np.arange(len(data[0]))

    fig, ax = plt.subplots(figsize=(4, 3))

    for i in range(num_data):
        label = labels[i]
        heights = data[i]
        color = colors[i]
        offset = (i - num_data / 2) * bar_width + bar_width / 2
        bar_positions = x + offset
        ax.bar(
            bar_positions,
            heights, bar_width,
            color=color,
            label=label
        )

    ax.set_ylabel('Action Frequency [%]', fontsize=LABEL_FONTSIZE - 1)
    ax.set_xlabel('Actions', fontsize=LABEL_FONTSIZE - 1)
    if x_labels is not None:
        assert len(x_labels) == len(data[0])
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    ax.tick_params(axis='both', labelsize=12)

    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncols=num_data,
        framealpha=1.0
    )
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor('lightgray')
    legend_frame.set_edgecolor('black')

    fig.tight_layout()
    # for handle in legend.legend_handles:
    #     handle.set_sizes([MARKER_SIZE_LEGEND])
    return fig, ax