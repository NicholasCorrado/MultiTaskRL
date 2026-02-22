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

def plot(x_dict, y_dict, linestyle_dict, color_dict):
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
        ylabel=f'Success Rate',
        labelsize='large',
        ticklabelsize='large',
        # linestyles=linestyle_dict,
        colors=color_dict,
        marker='',
    )
    plt.legend()
    plt.title(f'Average success rate over all tasks', fontsize='large')
    plt.ylim(0,1.1)

    # Use scientific notation for x-axis
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # set fontsize of scientific notation label
    ax.xaxis.get_offset_text().set_fontsize('large')

    plt.tight_layout()

if __name__ == "__main__":


    seaborn.set_theme(style='whitegrid')

    n_rows = 1
    n_cols = 1
    fig = plt.figure(figsize=(n_cols*9,n_rows*9))
    i = 1


    env_id = 'GridWorldEnv1_GridWorldEnv2_GridWorldEnv3_GridWorldEnv4'


    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}
    lr = 1e-2
    ns = 128

    ax = plt.subplot(n_rows, n_cols, i)
    i+=1

    # DRO AVERAGE ######################################################################################################
    # key = f"DRO with Return Gap"
    # results_dir = f"../../results/gridworld/return_ref/results/GridWorldEnv1_GridWorldEnv2_GridWorldEnv3_GridWorldEnv4/ppo/dro/lr_{lr}/ns_{ns}"
    color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))

    for i in range (2, 11):
        a = (i-1) / 10
        probs = [(1-a)/3, (1-a)/3, (1-a)/3, a]
        key = f"Hardest_Prob_={a}"
        results_dir = f"../../results/gridworld/{i}/results/GridWorldEnv1_GridWorldEnv2_GridWorldEnv3_GridWorldEnv4/ppo/no_dro"

        x, y = get_data(results_dir, y_name=f'success_rate')
        T = len(x) // 4
        if y is not None:
            x_dict[key] = x[:T]
            y_dict[key] = y[:, :T]
            linestyle_dict[key] = '-'
            color_dict[key] = next(color_palette)

    plot(x[:T], y_dict, linestyle_dict, color_dict)
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}

    plt.axhline(y=1, color='k', linestyle='--', label='Optimal success rate')
    plt.legend(fontsize='medium')

    save_dir = f'figures'
    save_name = f'success_rate_avg_ablate_for_diff_probs_init.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=200)

    plt.show()
