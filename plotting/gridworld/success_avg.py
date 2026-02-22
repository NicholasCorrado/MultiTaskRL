import os

import numpy as np
import seaborn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics
# from rliable.plot_utils import plot_sample_efficiency_curve
from plotting.utils import plot_sample_efficiency_curve

from plotting.utils import get_data

def plot(x_dict, y_dict, linestyle_dict, color_dict):
    results_dict = {algorithm: score for algorithm, score in y_dict.items()}
    aggr_func = lambda scores: np.array([metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
    scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=1000)

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
        linestyles=linestyle_dict,
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

    n_rows = 1
    n_cols = 1
    fig = plt.figure(figsize=(n_cols*4,n_rows*4))
    i = 1

    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}

    ax = plt.subplot(n_rows, n_cols, i)
    i+=1

    lr = 3e-3
    ns = 256
    path_dict = {
        'DRO': f"../chtc/results1/gw4/results/ppo/dro/lr_{lr}/ns_{ns}",
        'DRO, reweight': f"../chtc/results1/gw4/results/ppo/dro_reweight/lr_{lr}/ns_{ns}",
        'Hard First': f"../chtc/results1/gw4/results/ppo/hard_first/lr_{lr}/ns_{ns}",
        'Easy First': f"../chtc/results1/gw4/results/ppo/easy_first/lr_{lr}/ns_{ns}",
        'Learning Progress': f"../chtc/results1/gw4/results/ppo/learning_progress/lr_{lr}/ns_{ns}",
        'Uniform': f"../chtc/results1/gw4/results/ppo/uniform/lr_{lr}/ns_{ns}",
    }
    color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))

    for key, results_dir in path_dict.items():
            x, y = get_data(results_dir, y_name=f'success_rate')
            T = 100 # reduce this value if you want to truncate the data
            if y is not None:
                x_dict[key] = x[:T]
                y_dict[key] = y[:, :T]
                linestyle_dict[key] = '-'
                color_dict[key] = next(color_palette)

    plot(x_dict, y_dict, linestyle_dict, color_dict)
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}

    # plt.axhline(y=1, color='k', linestyle='--', label='Optimal\nsuccess rate')
    plt.axhline(y=1, color='k', linestyle='--')

    plt.legend(fontsize='large')
    # # Push plots down to make room for the the legend
    # fig.subplots_adjust(top=0.86)
    #
    # # Fetch and plot the legend from one of the subplots.
    # ax = fig.axes[0]
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', fontsize='large', ncols=2)

    save_dir = f'figures'
    save_name = f'success_rate_avg.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=200)

    plt.show()


