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

    ### PROPS ##################################################################################################
    from itertools import product
    from collections import namedtuple

    env_ids = ['Swimmer-v5', 'Hopper-v5', 'HalfCheetah-v5', 'Walker2d-v5', 'Ant-v5', 'Humanoid-v5']
    # Define parameters and their values in a dictionary
    params_dict = {
        # 'env_id': ['Swimmer-v5', 'Hopper-v5', 'HalfCheetah-v5', 'Walker2d-v5', 'Ant-v5', 'Humanoid-v5'],
        'lr': [1e-4, 3e-4, 1e-3],
        'ns': [1024, 2048, 4196, 8192],
    }
    # Create a namedtuple class with the parameter names
    params = namedtuple('params', params_dict.keys())

    data_dict = {}
    seaborn.set_theme(style='whitegrid')

    n_rows = 2
    n_cols = 3
    fig = plt.figure(figsize=(n_cols*3,n_rows*3))
    i = 1

    for env_id in env_ids:
        ax = plt.subplot(n_rows, n_cols, i)
        i+=1
        # Loop over parameter settings
        for values in product(*params_dict.values()):
            # Create a namedtuple instance for easy dot access to parameters
            p = params(*values)

            key = f"PPO"
            results_dir = f"../results/{env_id}/ppo/lr_{p.lr}/ns_{p.ns}"
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
    save_name = f'return_sweep.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')

    plt.show()
