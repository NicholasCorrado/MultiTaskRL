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

def plot(x_axis, y_dict, linestyle_dict, color_dict):
    results_dict = {algorithm: score for algorithm, score in y_dict.items()}
    aggr_func = lambda scores: np.array([metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
    scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=100)

    plot_sample_efficiency_curve(
        frames=x_axis,
        point_estimates=scores,
        interval_estimates=cis,
        ax=ax,
        algorithms=None,
        xlabel='Timestep',
        ylabel=f'Return',
        labelsize='large',
        ticklabelsize='large',
        # linestyles=linestyle_dict,
        colors=color_dict,
        marker='',
    )
    plt.legend()
    plt.title(f'Average return over all tasks', fontsize='large')
    # plt.ylim(-2.1,1.1)

    # Use scientific notation for x-axis
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # set fontsize of scientific notation label
    ax.xaxis.get_offset_text().set_fontsize('large')

    plt.tight_layout()

if __name__ == "__main__":


    seaborn.set_theme(style='whitegrid')

    n_rows = 1
    n_cols = 1
    fig = plt.figure(figsize=(n_cols*4,n_rows*4))
    i = 1


    env_id = 'PointMaze_UMaze-v3_PointMaze_Medium-v3_PointMaze_Large-v3'


    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    lr = 0.01
    ns = 128

    ax = plt.subplot(n_rows, n_cols, i)
    i+=1

    # PointMaze_UMaze AVERAGE ######################################################################################################
    key = f"PointMaze_UMaze"
    results_dir = f"../../results/pointmaze_singletask/results/PointMaze_UMaze-v3/ddpg"
    color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))

    x, y = get_data(results_dir, y_name=f'return')
    T = len(x)
    if y is not None:
        x_dict[key] = x[:T]
        y_dict[key] = y[:, :T]
        linestyle_dict[key] = '-'
        color_dict[key] = next(color_palette)

    # PointMaze_Medium-v3 AVERAGE #################################################################################################
    key = f"PointMaze_Medium-v3"
    results_dir = f"../../results/pointmaze_singletask/results/PointMaze_Medium-v3/ddpg"
    # color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))

    x, y = get_data(results_dir, y_name=f'return')
    T = len(x)
    if y is not None:
        x_dict[key] = x[:T]
        y_dict[key] = y[:, :T]
        linestyle_dict[key] = '-'
        color_dict[key] = next(color_palette)

    # PointMaze_Large-v3 AVERAGE #################################################################################################
    key = f"PointMaze_Large-v3"
    results_dir = f"../../results/pointmaze_singletask/results/PointMaze_Large-v3/ddpg"
    # color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))

    x, y = get_data(results_dir, y_name=f'return')
    T = len(x)
    if y is not None:
        x_dict[key] = x[:T]
        y_dict[key] = y[:, :T]
        linestyle_dict[key] = '-'
        color_dict[key] = next(color_palette)

    plot(x[:T], y_dict, linestyle_dict, color_dict)
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    plt.axhline(y=1, color='k', linestyle='--', label='Optimal return')
    plt.legend(fontsize='medium')
    # # Push plots down to make room for the the legend
    # fig.subplots_adjust(top=0.86)
    #
    # # Fetch and plot the legend from one of the subplots.
    # ax = fig.axes[0]
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', fontsize='large', ncols=2)


    save_dir = f'figures'
    save_name = f'single_task_return.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=200)

    plt.show()
