import os

import numpy as np
import seaborn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics
# from rliable.plot_utils import plot_sample_efficiency_curve
from utils import plot_sample_efficiency_curve

from utils import get_data

def plot(x_dict, y_dict, linestyle_dict, color_dict, ylabel, title):
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
        linestyles=linestyle_dict,
        colors=color_dict,
        marker='',
    )
    plt.legend()
    plt.title(title, fontsize='large')
    plt.ylim(0,1.1)

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


    env_id = 'Bandit1-v0_Bandit2-v0_Bandit3-v0_Bandit4-v0_Bandit5-v0'


    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    lr = 1e-2
    ns = 128

    # DRO TASK i #######################################################################################################
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    ax = plt.subplot(n_rows, n_cols, subplot_i)
    subplot_i+=1

    results_dir = f"../chtc/results/bandit_ablate/results/{env_id}/ppo/dro/success_ref/lr_{lr}/ns_{ns}"
    color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))
    for i in range(5):
        key = f"Task {i+1}"
        x, y = get_data(results_dir, y_name=f'success_rate_{i}')

        T = len(x)//1
        if y is not None:
            x_dict[key] = x[:T]
            y_dict[key] = y[:, :T]
            linestyle_dict[key] = '-'
            color_dict[key] = next(color_palette)
    plot(x_dict, y_dict, linestyle_dict, color_dict, title='Task Success Rate:\nDRO with Success Gap', ylabel='Success Rate')
    plt.axhline(y=1, color='k', linestyle='--', label='Optimal\nsuccess rate')
    plt.legend()

    # STANDARD TASK i ##################################################################################################
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    ax = plt.subplot(n_rows, n_cols, subplot_i)
    subplot_i+=1


    results_dir = f"../chtc/results/bandit_ablate2/results/{env_id}/ppo/dro/return_ref/lr_{lr}/ns_{ns}"
    color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))
    for i in range(5):
        key = f"Task {i+1}"
        x, y = get_data(results_dir, y_name=f'success_rate_{i}')

        T = len(x)//1
        if y is not None:
            x_dict[key] = x[:T]
            y_dict[key] = y[:, :T]
            linestyle_dict[key] = '-'
            color_dict[key] = next(color_palette)
    plot(x_dict, y_dict, linestyle_dict, color_dict, title='Task Success Rate:\nDRO with Return Gap', ylabel='Success Rate')
    plt.axhline(y=1, color='k', linestyle='--', label='Optimal\nsuccess rate')
    plt.legend()


    # DRO TASK i #######################################################################################################
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    ax = plt.subplot(n_rows, n_cols, subplot_i)
    subplot_i+=1

    results_dir = f"../chtc/results/bandit_ablate/results/{env_id}/ppo/dro/success_ref/lr_{lr}/ns_{ns}"
    color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))
    for i in range(5):
        key = f"Task {i+1}"
        x, y = get_data(results_dir, y_name=f'return_{i}')

        T = len(x)//1
        if y is not None:
            x_dict[key] = x[:T]
            y_dict[key] = y[:, :T]
            linestyle_dict[key] = '-'
            color_dict[key] = next(color_palette)
    plot(x_dict, y_dict, linestyle_dict, color_dict, title='Task Return:\nDRO with Success Gap', ylabel='Return')
    plt.axhline(y=1, color='k', linestyle='--', label='Optimal return')
    plt.legend()

    # STANDARD TASK i ##################################################################################################
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    ax = plt.subplot(n_rows, n_cols, subplot_i)
    subplot_i+=1

    results_dir = f"../chtc/results/bandit_ablate2/results/{env_id}/ppo/dro/return_ref/lr_{lr}/ns_{ns}"
    color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))
    for i in range(5):
        key = f"Task {i+1}"
        x, y = get_data(results_dir, y_name=f'return_{i}')

        T = len(x)//1
        if y is not None:
            x_dict[key] = x[:T]
            y_dict[key] = y[:, :T]
            linestyle_dict[key] = '-'
            color_dict[key] = next(color_palette)
    plot(x_dict, y_dict, linestyle_dict, color_dict, title='Task Return:\nDRO with Return Gap', ylabel='Return')
    plt.axhline(y=1, color='k', linestyle='--', label='Optimal return')
    plt.legend()

    #
    # DRO TASK i #######################################################################################################
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    ax = plt.subplot(n_rows, n_cols, subplot_i)
    subplot_i+=1

    results_dir = f"../chtc/results/bandit_ablate/results/{env_id}/ppo/dro/success_ref/lr_{lr}/ns_{ns}"
    color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))
    for i in range(5):
        key = f"Task {i+1}"
        x, y = get_data(results_dir, y_name=f'task_probs_{i}')

        T = len(x)//1
        if y is not None:
            x_dict[key] = x[:T]
            y_dict[key] = y[:, :T]
            linestyle_dict[key] = '-'
            color_dict[key] = next(color_palette)

    plot(x_dict, y_dict, linestyle_dict, color_dict, title='Sampling Weights:\nDRO with Success Gap', ylabel='Sampling Weight')

    # STANDARD TASK i ##################################################################################################
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    ax = plt.subplot(n_rows, n_cols, subplot_i)
    subplot_i+=1

    results_dir = f"../chtc/results/bandit_ablate2/results/{env_id}/ppo/dro/return_ref/lr_{lr}/ns_{ns}"
    color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))
    for i in range(5):
        key = f"Task {i+1}"
        x, y = get_data(results_dir, y_name=f'task_probs_{i}')

        T = len(x)//1
        if y is not None:
            x_dict[key] = x[:T]
            y_dict[key] = y[:, :T]
            linestyle_dict[key] = '-'
            color_dict[key] = next(color_palette)

    plot(x_dict, y_dict, linestyle_dict, color_dict, title='Sampling Weights:\nDRO with Return Gap', ylabel='Sampling Weight')
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}




    # Push plots down to make room for the the legend
    # fig.subplots_adjust(top=0.80)
    #
    # # Fetch and plot the legend from one of the subplots.
    # ax = fig.axes[0]
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', fontsize='large', ncols=2)
    plt.tight_layout()

    save_dir = f'figures'
    save_name = f'success_rate_and_weights_ablate.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=200)

    plt.show()
