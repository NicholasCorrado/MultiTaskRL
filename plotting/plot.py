import os

import numpy as np
import seaborn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics
from rliable.plot_utils import plot_sample_efficiency_curve

from utils import get_data

if __name__ == "__main__":

    data_dict = {}
    seaborn.set_theme(style='whitegrid')

    n_rows = 3
    n_cols = 3
    fig = plt.figure(figsize=(n_cols*3,n_rows*3))
    i = 1

    env_ids = ['Task_1.0', 'Task_2.0', 'Task_3.0', 'Task_4.0', 'Task_5.0', 'Task_6.0', 'Task_7.0', 'Task_8.0', 'Task_9.0']

    for env_id in env_ids:
        key = f"PPO"
        results_dir = f"../results/{env_id}/ppo/"
        if not os.path.exists(results_dir):
            print (f'Task {env_id} does not have result!')
            continue

        ax = plt.subplot(n_rows, n_cols, i)
        ax.set_title(env_id)
        i+=1

        # Now we can use dot notation which is much cleaner
        x, y = get_data(results_dir, x_name='timestep', y_name='return', filename='evaluations.npz')
        if y is not None:
            data_dict[key] = y

        results_dict = {algorithm: score for algorithm, score in data_dict.items()}
        aggr_func = lambda scores: np.array([metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
        scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=1000)

        plot_sample_efficiency_curve(
            frames=x,
            point_estimates=scores,
            interval_estimates=cis,
            ax=ax,
            algorithms=None,
            xlabel='Timestep',
            ylabel=f'Return',
            # title=f'{env_id}',
            labelsize='large',
            ticklabelsize='large',
        )
        # Use scientific notation for x-axis
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # set fontsize of scientific notation label
        ax.xaxis.get_offset_text().set_fontsize('large')

        # Set log scale
        # plt.xscale('log')
        # plt.yscale('log')

        plt.tight_layout()

    # Push plots down to make room for the the legend
    fig.subplots_adjust(top=0.88)

    # Fetch and plot the legend from one of the subplots.
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize='large')

    save_dir = f'figures'
    save_name = f'return.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')

    plt.show()
