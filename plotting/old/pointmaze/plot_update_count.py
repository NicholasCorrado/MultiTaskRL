import os

import numpy as np
import seaborn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics
# from rliable.plot_utils import plot_sample_efficiency_curve
from rliable.plot_utils import plot_sample_efficiency_curve

from utils import get_data

def plot(x_dict, y_dict, linestyle_dict, color_dict, ylabel, title, y_bottom=0, y_top=1.1):
    results_dict = {algorithm: score for algorithm, score in y_dict.items()}
    aggr_func = lambda scores: np.array([metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
    scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=100)

    plot_sample_efficiency_curve(
        frames=x_dict,
        point_estimates=scores,
        interval_estimates=cis,
        ax=ax,
        algorithms=None,
        xlabel='Timestep',
        ylabel=ylabel,
        labelsize='large',
        ticklabelsize='large',
        # linestyles=linestyle_dict,
        colors=color_dict,
        marker='',
    )
    plt.legend()
    # plt.ylim(y_bottom,y_top)
    plt.title(title, fontsize='large')

    # Use scientific notation for x-axis
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # set fontsize of scientific notation label
    ax.xaxis.get_offset_text().set_fontsize('large')

    # plt.tight_layout()

if __name__ == "__main__":


    seaborn.set_theme(style='whitegrid')

    n_rows = 3
    n_cols = 2
    fig = plt.figure(figsize=(n_cols*4,n_rows*4))
    subplot_i = 1

    n = 3 # number of environments

    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    lr = 1e-2
    ns = 128

    C = 1

    task_name = "ppo_pointmaze_2"
    sub_dir_name_standard, sub_dir_name_dro = "no_dro", "success_ref"
    env_id = 'PointMaze1_PointMaze2_PointMaze3'

    # DRO TASK i #######################################################################################################
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    ax = plt.subplot(n_rows, n_cols, subplot_i)
    subplot_i+=1

    results_dir = f"../../results/{task_name}/{sub_dir_name_dro}/results/{env_id}/ppo"
    color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))
    for i in range(n):
        key = f"Task {i+1}"
        x, y = get_data(results_dir, y_name=f'update')
        print(y)
        exit(0)
        y = y[:, :, i]
        print(y)

        T = (int)(len(x) / C)
        if y is not None:
            x_dict[key] = x[:T]
            y_dict[key] = y[:, :T]
            linestyle_dict[key] = '-'
            color_dict[key] = next(color_palette)
    plot(x[:T], y_dict, linestyle_dict, color_dict, title='Task Success Rate: DRO', ylabel='Success Rate')
    plt.axhline(y=1, color='k', linestyle='--', label='Optimal\nsuccess rate')
    plt.legend()

    # STANDARD TASK i ##################################################################################################
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    ax = plt.subplot(n_rows, n_cols, subplot_i)
    subplot_i+=1

    results_dir = f"../../results/{task_name}/{sub_dir_name_standard}/results/{env_id}/ppo"
    color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))
    for i in range(n):
        key = f"Task {i+1}"
        x, y = get_data(results_dir, y_name=f'update')
        y = y[:, :, i]

        T = (int)(len(x) / C)
        if y is not None:
            x_dict[key] = x[:T]
            y_dict[key] = y[:, :T]
            linestyle_dict[key] = '-'
            color_dict[key] = next(color_palette)
    plot(x[:T], y_dict, linestyle_dict, color_dict, title='Task Success Rate: Standard', ylabel='Success Rate')
    plt.axhline(y=1, color='k', linestyle='--', label='Optimal\nsuccess rate')
    plt.legend()

    # Push plots down to make room for the the legend
    # fig.subplots_adjust(top=0.80)
    #
    # # Fetch and plot the legend from one of the subplots.
    # ax = fig.axes[0]
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', fontsize='large', ncols=2)
    plt.tight_layout()

    save_dir = f'figures'
    save_name = f'success_rate_and_weights.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=200)

    plt.show()
