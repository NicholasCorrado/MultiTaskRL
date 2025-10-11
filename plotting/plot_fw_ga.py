import os
from logging import exception

import numpy as np
import seaborn

import matplotlib
from sympy.printing.pretty.pretty_symbology import line_width

from demo import env_id

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics
from rliable.plot_utils import plot_sample_efficiency_curve
import seaborn as sns

from utils import get_data

if __name__ == "__main__":

    data_dict = {}
    seaborn.set_theme(style='whitegrid')

    env_id_list = [['FixedWeightTest/results/BanditEasy-v0_BanditHard-v0/learning_rate=0.001/w_[0.5, 0.5]',
                    'DROTest/results/BanditEasy_BanditHard/learning_rate=0.001/w_lr=1.0'],
                   ['FixedWeightTest/results/BanditEasy-v0_BanditHard-v0/learning_rate=0.0003/w_[0.5, 0.5]',
                    'DROTest/results/BanditEasy_BanditHard/learning_rate=0.0003/w_lr=1.0'],
                   ['FixedWeightTest/results/BanditEasy-v0_BanditHard-v0/learning_rate=0.0001/w_[0.5, 0.5]',
                    'DROTest/results/BanditEasy_BanditHard/learning_rate=0.0001/w_lr=1.0'],
                   ]

    n_rows = 2
    n_cols = len(env_id_list)
    fig = plt.figure(figsize=(18 * n_cols,36))
    i = 1


    for env_ids_l in env_id_list:

        ax = plt.subplot(n_rows, n_cols, i)

        j = 1

        for env_id in env_ids_l:

            key = f"PPO Average: {env_id}"
            results_dir = f"../results/{env_id}/ppo/"
            if not os.path.exists(results_dir):
                print (f'Task {env_id} does not have result!')
                continue


            x, y, env_ids, task_ids = get_data(results_dir, x_name='timestep', y_name='success_rate',
                                               filename='evaluations.npz')

            if y is None:
                continue

            if j == 1:
                marker = 'v'
            else:
                marker = 's'

            if y is not None:
                data_dict[key] = y

            num_envs = len(env_ids)  # Number of environments
            num_timesteps = len(x)  # Number of x

            env_returns = [[] for _ in range(num_envs)]
            env_returns_avged = [[] for _ in range(num_envs)]

            # Organizing data per environment
            for t in range(num_timesteps):
                for env in range(num_envs):
                    env_returns[env].append([y[i][t][0][env] for i in range(len(y))])
                    env_returns_avged[env].append(np.mean([y[i][t][0][env] for i in range(len(y))]))

            if len(env_ids) > 1:

                data_dict = {}

                if y is not None:
                    data_dict[key] = y

                results_dict = {algorithm: score for algorithm, score in data_dict.items()}
                aggr_func = lambda scores: np.array(
                    [metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
                scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=1000)

                if j == 1:
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
                else:
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
                        colors = dict(zip([key], ['y']))
                    )

            data_dict = {}

            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))


            env_id_num = 0

            # Plot each environment's results
            for env_idx in range(len(env_returns)):
                key = f"PPO Task {env_id} {task_ids[env_idx]}: {env_ids[env_idx]}"
                env_id_num += 1

                data_dict = {}
                data_dict[key] = np.array(env_returns[env_idx]).transpose()
                results_dict = {algorithm: score for algorithm, score in data_dict.items()}

                aggr_func = lambda scores: np.array(
                    [metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
                scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=1000)

                if env_id_num == 1:
                    ax.plot(x, scores[key], label = key, linewidth=1.0, marker = marker, color  = 'r')
                else:
                    ax.plot(x, scores[key], label = key, linewidth=1.0, marker = marker, color = 'g')
                ax.fill_between(x, cis[key][0], cis[key][1], alpha = 0.2)

            j += 1

        # Set log scale
        # plt.xscale('log')
        # plt.yscale('log')
        ax.legend(loc="upper left", fontsize="small")
        # ax.set_ylim(bottom=0.7, top=1.1)
        i += 1

    for env_ids in env_id_list:

        env_id = env_ids[1]

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
        ax.set_ylim(bottom=-0.1, top=1.1)

    plt.tight_layout()

    # Push plots down to make room for the the legend
    fig.subplots_adjust(top=0.88)


    save_dir = f'figures'
    save_name = f'return_double_weight_ppo.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')

    plt.show()
