import os

import numpy as np
import seaborn

import matplotlib
from sympy.printing.pretty.pretty_symbology import line_width

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics
from rliable.plot_utils import plot_sample_efficiency_curve

from utils import get_data

if __name__ == "__main__":

    data_dict = {}
    seaborn.set_theme(style='whitegrid')

    n_rows = 1
    n_cols = 3
    fig = plt.figure(figsize=(27,9))
    i = 1

    env_ids = ['BanditEasy', 'BanditHard', 'BanditEasy_BanditHard']

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
        x, y, env_ids, task_ids = get_data(results_dir, x_name='timestep', y_name='return_avg', filename='evaluations.npz')
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

        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # individual record for each environment
        timesteps, returns, env_ids, task_ids = get_data(results_dir, x_name='timestep', y_name='returns')
        returns = returns[0]

        if isinstance(returns, float):
            ax.plot(timesteps, returns, label='Mean Return')
        else:  # Handling return_list
            num_envs = len(returns[0])  # Number of environments
            num_timesteps = len(timesteps)  # Number of timesteps

            env_returns = [[] for _ in range(num_envs)]

            # Organizing data per environment
            for t in range(num_timesteps):
                for env in range(num_envs):
                    env_returns[env].append(returns[t][env])

            # Plot each environment's results
            for env_idx, env_data in enumerate(env_returns):
                ax.plot(timesteps, env_data, label=f'Task {task_ids[env_idx]}: {env_ids[env_idx]}',linewidth=1.0)

        # Set log scale
        # plt.xscale('log')
        # plt.yscale('log')
        ax.legend(loc="upper left", fontsize="small")

    plt.tight_layout()

    # Push plots down to make room for the the legend
    fig.subplots_adjust(top=0.88)


    save_dir = f'figures'
    save_name = f'return_noavg.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')

    plt.show()
