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


def first_reach_threshold_updates(x, y, threshold=0.9):
    """For each run, return first x where success_rate reaches threshold."""
    first_updates = []
    for run_success in y:
        reached = np.where(run_success >= threshold)[0]
        if reached.size > 0:
            first_updates.append(x[reached[0]])
    return np.array(first_updates, dtype=float)


def plot(x_dict, y_dict, linestyle_dict, color_dict, x_name = 'dro_eta'):
    results_dict = {algorithm: score for algorithm, score in y_dict.items()}
    aggr_func = lambda scores: np.array([metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
    scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=1000)

    plot_sample_efficiency_curve(
        frames=x_dict,
        point_estimates=scores,
        interval_estimates=cis,
        ax=ax,
        algorithms=None,
        xlabel=x_name,
        ylabel='Timestep at first 90% success',
        labelsize='large',
        ticklabelsize='large',
        linestyles=linestyle_dict,
        colors=color_dict,
        marker='',
    )
    # plt.legend()
    # plt.xticks(rotation=45, ha='right')
    # plt.xticks(fontsize=10)
    # plt.title('First update reaching 90% success', fontsize='large')

    # Use scientific notation for y-axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize('large')

    plt.tight_layout()

if __name__ == "__main__":

    n_rows = 1
    n_cols = 1
    fig = plt.figure(figsize=(n_cols*5,n_rows*5))
    i = 1

    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}

    ax = plt.subplot(n_rows, n_cols, i)
    i+=1

    lr = 3e-3
    ns = 256

    path_dict = {}

    key = 'dro/dro_eta'
    x_name = 'dro_eta'
    # vals = [1, 2, 4, 8, 16, 32, 64]
    vals = [0, 1, 2, 3, 4, 5, 6]
    for i in vals:
        path_dict[f'f={i}'] = f'../../results/exp_1/dro_eta={(2 ** i)}/results/ppo/dro/lr_0.003/ns_256'

    # key = 'dro/dro_step_size'
    # x_name = 'dro_step_size'
    # vals = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    # for i in vals:
    #     path_dict[f'f={i}'] = f'../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_{i}'

    # key = 'dro/dro_num_steps'
    # x_name = rf'$log_2$(dro_num_steps)'
    # # vals = [16, 32, 64, 128, 256, 512, 1024]
    # vals = [4, 5, 6, 7, 8, 9, 10]
    # for i in vals:
    #     path_dict[f'f={i}'] = f'../../results/exp_3_1/success_ref/results/ppo/dro/dro_num_steps_{(2 ** i)}'

    # key = 'dro/num_steps'
    # x_name = rf'$log_2$(num_steps)'
    # # vals = [256, 512, 1024, 2048]
    # vals = [8, 9, 10, 11]
    # for i in vals:
    #     path_dict[f'f={i}'] = f'../../results/exp_4_1/success_ref/results/ppo/dro/num_steps_{(2 ** i)}'

    # key = 'learning_progress/num_steps'
    # x_name = key
    # vals = [256, 512, 1024, 2048]
    # for i in vals:
    #     path_dict[f'f={i}'] = f'../../results/exp_4_1/success_ref/results/ppo/learning_progress/num_steps_{i}'

    # key = 'uniform/num_steps'
    # x_name = key
    # vals = [256, 512, 1024, 2048]
    # for i in vals:
    #     path_dict[f'f={i}'] = f'../../results/exp_4_1/success_ref/results/ppo/uniform/num_steps_{i}'

    # key = 'learning_progress/dro_eta'
    # x_name = rf'$log_2(dro_eta)$'
    # # vals = [1, 2, 4, 8, 16, 32, 64]
    # vals = [0, 1, 2, 3, 4, 5, 6]
    # for i in vals:
    #     path_dict[f'f={i}'] = f'../../results/exp_lp_dro_eta/dro_eta={(2 ** i)}/results/ppo/learning_progress/dro_eta_0.2'

    # key = 'learning_progress/dro_num_steps'
    # x_name = rf'$log_2$(dro_num_steps)'
    # # vals = [16, 32, 64, 128, 256, 512, 1024]
    # vals = [4, 5, 6, 7, 8, 9, 10]
    # for i in vals:
    #     path_dict[f'f={i}'] = f'../../results/exp_lp_dro_ns/success_ref/results/ppo/learning_progress/dro_num_steps_{(2 ** i)}'

    # key = 'learning_progress/dro_step_size'
    # x_name = 'dro_step_size'
    # vals = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    # for i in vals:
    #     path_dict[f'f={i}'] = f'../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_{i}'

    per_val_first_updates = []
    valid_vals = []

    for val in vals:
        results_dir = path_dict[f'f={val}']
        x, y = get_data(results_dir, x_name='timestep', y_name='success_rate')
        if x is None or y is None:
            continue

        T = len(x)
        updates = first_reach_threshold_updates(x[:T], y[:, :T], threshold=0.9)
        if updates.size == 0:
            print(f"{key}={val}: no run reached 90% success")
            continue

        valid_vals.append(val)
        per_val_first_updates.append(updates)
        print(f"{key}={val}: reached 90% in {updates.size}/{y.shape[0]} runs")

    if len(per_val_first_updates) == 0:
        raise ValueError("No valid runs reached 90% success in any task.")

    # Keep a common run count so y has shape [num_runs, num_task_vals].
    # This allows using the existing rliable-based CI plotting pipeline.
    common_runs = min(arr.size for arr in per_val_first_updates)
    if common_runs < 2:
        print("Warning: fewer than 2 common runs; confidence interval may be unstable.")

    first_update_matrix = np.stack([arr[:common_runs] for arr in per_val_first_updates], axis=1)

    method_name = key
    x_dict[method_name] = np.array(valid_vals, dtype=float)
    y_dict[method_name] = first_update_matrix
    linestyle_dict[method_name] = '-'
    color_dict[method_name] = seaborn.color_palette('colorblind', n_colors=1)[0]

    plot(x_dict, y_dict, linestyle_dict, color_dict, x_name)
    x_dict, y_dict, linestyle_dict, color_dict = {}, {}, {}, {}

    plt.xticks(valid_vals)
    plt.grid(alpha=0.25)
    # plt.legend(fontsize='large')
    # # Push plots down to make room for the the legend
    # fig.subplots_adjust(top=0.86)
    #
    # # Fetch and plot the legend from one of the subplots.
    # ax = fig.axes[0]
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', fontsize='large', ncols=2)

    save_dir = f'figures'
    save_name = f'{key}_min_steps_to_90.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=200)

    plt.show()


