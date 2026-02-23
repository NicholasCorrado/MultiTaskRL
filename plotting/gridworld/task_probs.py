import os
import numpy as np
import seaborn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics
from plotting.utils import plot_sample_efficiency_curve
from plotting.utils import get_data


def plot_metric_on_ax(ax, path_dict, linestyle_dict, color_dict, metric_name, title):
    x_dict, y_dict = {}, {}

    for key, results_dir in path_dict.items():
        x, y = get_data(results_dir, y_name=metric_name)
        if y is not None:
            x_dict[key] = x
            y_dict[key] = y

    results_dict = {algorithm: score for algorithm, score in y_dict.items()}
    aggr_func = lambda scores: np.array(
        [metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])]
    )
    scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=100)

    plot_sample_efficiency_curve(
        frames=x_dict,
        point_estimates=scores,
        interval_estimates=cis,
        ax=ax,
        algorithms=None,
        xlabel='Timestep',
        ylabel='Probability',
        labelsize='large',
        ticklabelsize='large',
        linestyles=linestyle_dict,
        colors=color_dict,
        marker='',
    )

    ax.set_title(title, fontsize='large')
    ax.set_ylim(0, 1.05)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize('large')


if __name__ == "__main__":

    fig = plt.figure(figsize=(16,4))

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
    palette = seaborn.color_palette('colorblind', n_colors=10)
    color_dict = dict(zip(path_dict.keys(), palette))
    linestyle_dict = {k: '-' for k in path_dict.keys()}

    for task_i in range(4):
        ax = plt.subplot(1,4,task_i+1)
        plot_metric_on_ax(
            ax,
            path_dict,
            linestyle_dict,
            color_dict,
            metric_name=f'task_probs_{task_i}',
            title=f'Task {task_i} sampling prob'
        )

        if task_i == 0:
            ax.legend(fontsize='large')

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/task_probs.png", dpi=200)
    plt.show()