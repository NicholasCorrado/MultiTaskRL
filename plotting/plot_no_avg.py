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

    n_rows = 3
    n_cols = 3
    fig = plt.figure(figsize=(n_cols*3,n_rows*3))
    i = 1

    env_ids = ['Task_1.0', 'Task_2.0', 'Task_3.0', 'Task_4.0', 'Task_5.0', 'Task_6.0', 'Task_7.0', 'Task_8.0', 'Task_9.0']

    for env_id in env_ids:
        results_dir = f"../results/{env_id}/ppo/"
        if not os.path.exists(results_dir):
            print (f'Task {env_id} does not have result!')
            continue

        ax = plt.subplot(n_rows, n_cols, i)
        ax.set_title(env_id)
        i+=1

        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # individual record for each environment
        timesteps, returns, env_ids = get_data(results_dir, x_name='timestep', y_name='returns')
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
                ax.plot(timesteps, env_data, label=env_ids[env_idx],linewidth=1.0)

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
