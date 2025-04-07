import os
from logging import exception

import numpy as np
import seaborn

import matplotlib
from sympy.printing.pretty.pretty_symbology import line_width
from sympy.simplify.simplify import bottom_up

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics
from rliable.plot_utils import plot_sample_efficiency_curve

from utils import get_data

if __name__ == "__main__":

    data_dict = {}
    seaborn.set_theme(style='whitegrid')

    n_rows = 2
    n_cols = 3
    fig = plt.figure(figsize=(27,18))
    i = 1

    env_id_list = ['BanditEasy-v0_BanditHard-v0/w_lr=1.0',
                   'BanditEasy-v0_BanditHard-v0/w_lr=0.1',
                   'BanditEasy-v0_BanditHard-v0/w_lr=0.01',
                   ]

    for env_id in env_id_list:
        # try:
            # Now we can use dot notation which is much cleaner

        key = f"PPO Average"
        results_dir = f"../results/{env_id}/ppo/"
        if not os.path.exists(results_dir):
            print (f'Task {env_id} does not have result!')
            continue

        ax = plt.subplot(n_rows, n_cols, i)
        ax.set_title(env_id)
        i+=1

        x, y, env_ids, task_ids = get_data(results_dir, x_name='timestep', y_name='return_avg', filename='evaluations.npz')

        if len(env_ids) > 1:

            data_dict = {}

            if y is not None:
                data_dict[key] = y

            results_dict = {algorithm: score for algorithm, score in data_dict.items()}
            aggr_func = lambda scores: np.array(
                [metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
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

        data_dict = {}

        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        if y is None:
            continue

        x, y, env_ids, task_ids = get_data(results_dir, x_name='timestep', y_name='returns',
                                           filename='evaluations.npz')
        if y is not None:
            data_dict[key] = y

        num_envs = len(env_ids)  # Number of environments
        num_timesteps = len(x)  # Number of x

        env_returns = [[] for _ in range(num_envs)]

        # Organizing data per environment
        for t in range(num_timesteps):
            for env in range(num_envs):
                env_returns[env].append([y[i][t][env] for i in range(len(y))])

        # Plot each environment's results
        for env_idx in range(len(env_returns)):
            key = f"PPO Task {task_ids[env_idx]}: {env_ids[env_idx]}"

            data_dict = {}
            data_dict[key] = np.array(env_returns[env_idx]).transpose()
            results_dict = {algorithm: score for algorithm, score in data_dict.items()}

            aggr_func = lambda scores: np.array(
                [metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
            scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=1000)

            ax.plot(x, scores[key], label = key, linewidth=1.0)
            ax.fill_between(x, cis[key][0], cis[key][1], alpha = 0.2)

        # Set log scale
        # plt.xscale('log')
        # plt.yscale('log')
        ax.legend(loc="upper left", fontsize="small")
        ax.set_ylim(top=1.5)
        ax.set_ylim(bottom=0.0, top=1.5)


    for env_id in env_id_list:

        key = f"PPO Average"
        results_dir = f"../results/{env_id}/ppo/"
        if not os.path.exists(results_dir):
            print (f'Task {env_id} does not have result!')
            continue

        ax = plt.subplot(n_rows, n_cols, i)
        ax.set_title(env_id)
        i+=1

        x, y, env_ids, task_ids = get_data(results_dir, x_name='timestep', y_name='weights',
                                           filename='evaluations.npz')
        if y is not None:
            data_dict[key] = y

        num_envs = len(env_ids)  # Number of environments
        num_timesteps = len(x)  # Number of x

        env_returns = [[] for _ in range(num_envs)]

        # Organizing data per environment
        for t in range(num_timesteps):
            for env in range(num_envs):
                env_returns[env].append([y[i][t][env] for i in range(len(y))])

        # Plot each environment's results
        for env_idx in range(len(env_returns)):
            key = f"Weight for PPO Task {task_ids[env_idx]} {env_ids[env_idx]}"

            data_dict = {}
            data_dict[key] = np.array(env_returns[env_idx]).transpose()
            results_dict = {algorithm: score for algorithm, score in data_dict.items()}

            aggr_func = lambda scores: np.array(
                [metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
            scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=1000)

            ax.plot(x, scores[key], label = key, linewidth=1.0)
            ax.fill_between(x, cis[key][0], cis[key][1], alpha = 0.2)

        # Set log scale
        # plt.xscale('log')
        # plt.yscale('log')
        ax.legend(loc="upper left", fontsize="small")
        ax.set_ylim(bottom=0.0, top=1.5)

    plt.tight_layout()

    # Push plots down to make room for the the legend
    fig.subplots_adjust(top=0.88)


    save_dir = f'figures'
    save_name = f'return_change_weight_ppo.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')

    plt.show()
