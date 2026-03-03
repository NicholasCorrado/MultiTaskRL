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
        xlabel='Update',
        ylabel=f'Success Rate',
        labelsize='large',
        ticklabelsize='large',
        linestyles=linestyle_dict,
        colors=color_dict,
        marker='',
    )
    plt.legend()
    plt.title(f'Learning Progress \n - \n Average success rate over all tasks', fontsize='large')
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
    path_dict = {}

    # save_title = 'dro/dro_eta'
    # for i in [1, 2, 4, 8, 16, 32, 64]:
    #     path_dict[f'dro/dro_eta={i}'] = f'../../results/exp_1/dro_eta={i}/results/ppo/dro/lr_0.003/ns_256'

    # save_title = 'dro/dro_step_size'
    # for i in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    #     path_dict[f'dro/dro_step_size={i}'] = f'../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_{i}'

    # save_title = 'dro/dro_num_steps'
    # for i in [16, 32, 64, 128, 256, 512, 1024]:
    #     path_dict[f'dro/dro_num_steps={i}'] = f'../../results/exp_3_1/success_ref/results/ppo/dro/dro_num_steps_{i}'

    # save_title = 'dro/num_steps'
    # for i in [256, 512, 1024, 2048]:
    #     path_dict[f'dro/num_steps={i}'] = f'../../results/exp_4_1/success_ref/results/ppo/dro/num_steps_{i}'

    # save_title = 'learning_progress/num_steps'
    # for i in [256, 512, 1024, 2048]:
    #     path_dict[f'learning_progress/num_steps={i}'] = f'../../results/exp_4_1/success_ref/results/ppo/learning_progress/num_steps_{i}'

    # save_title = 'uniform/num_steps'
    # for i in [256, 512, 1024, 2048]:
    #     path_dict[f'uniform/num_steps={i}'] = f'../../results/exp_4_1/success_ref/results/ppo/uniform/num_steps_{i}'

    # save_title = 'learning_progress/dro_eta'
    # for i in [1, 2, 4, 8, 16, 32, 64]:
    #     path_dict[f'learning_progress/dro_eta={i}'] = f'../../results/exp_lp_dro_eta/dro_eta={i}/results/ppo/learning_progress/dro_eta_0.2'

    # save_title = 'learning_progress/dro_num_steps'
    # for i in [16, 32, 64, 128, 256, 512, 1024]:
    #     path_dict[f'learning_progress/dro_num_steps={i}'] = f'../../results/exp_lp_dro_ns/success_ref/results/ppo/learning_progress/dro_num_steps_{i}'

    save_title = 'learning_progress/dro_step_size'
    for i in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        path_dict[f'learning_progress/dro_step_size={i}'] = f'../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_{i}'


    # path_dict = {
    #     # 'DRO': f"../chtc/results1/gw4/results/ppo/dro/lr_{lr}/ns_{ns}",
    #     # 'DRO, reweight': f"../chtc/results1/gw4/results/ppo/dro_reweight/lr_{lr}/ns_{ns}",
    #     # 'Hard First': f"../chtc/results1/gw4/results/ppo/hard_first/lr_{lr}/ns_{ns}",
    #     # 'Easy First': f"../chtc/results1/gw4/results/ppo/easy_first/lr_{lr}/ns_{ns}",
    #     # 'Learning Progress': f"../chtc/results1/gw4/results/ppo/learning_progress/lr_{lr}/ns_{ns}",
    #     # 'Uniform': f"../chtc/results1/gw4/results/ppo/uniform/lr_{lr}/ns_{ns}",
    #
    #     # 'num_steps=256': f"../../results/exp_4_1/success_ref/results/ppo/uniform/num_steps_256",
    #     # 'num_steps=512': f"../../results/exp_4_1/success_ref/results/ppo/uniform/num_steps_512",
    #     # 'num_steps=1024': f"../../results/exp_4_1/success_ref/results/ppo/uniform/num_steps_1024",
    #     # 'num_steps=2048': f"../../results/exp_4_1/success_ref/results/ppo/uniform/num_steps_2048",
    #
    #     # 'dro_step_size=0.01': f"../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.01",
    #     # 'dro_step_size=0.05': f"../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.05",
    #     # 'dro_step_size=0.1': f"../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.1",
    #     # 'dro_step_size=0.2': f"../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.2",
    #     # 'dro_step_size=0.4': f"../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.4",
    #     # 'dro_step_size=0.6': f"../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.6",
    #     # 'dro_step_size=0.8': f"../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.8",
    #     # 'dro_step_size=1.0': f"../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_1.0",
    #
    #     # 'dro_eta=1': f"../../results/exp_lp_dro_eta/dro_eta=1/results/ppo/learning_progress/dro_eta_0.2",
    #     # 'dro_eta=2': f"../../results/exp_lp_dro_eta/dro_eta=2/results/ppo/learning_progress/dro_eta_0.2",
    #     # 'dro_eta=4': f"../../results/exp_lp_dro_eta/dro_eta=4/results/ppo/learning_progress/dro_eta_0.2",
    #     # 'dro_eta=8': f"../../results/exp_lp_dro_eta/dro_eta=8/results/ppo/learning_progress/dro_eta_0.2",
    #     # 'dro_eta=16': f"../../results/exp_lp_dro_eta/dro_eta=16/results/ppo/learning_progress/dro_eta_0.2",
    #     # 'dro_eta=32': f"../../results/exp_lp_dro_eta/dro_eta=32/results/ppo/learning_progress/dro_eta_0.2",
    #     # 'dro_eta=64': f"../../results/exp_lp_dro_eta/dro_eta=64/results/ppo/learning_progress/dro_eta_0.2",
    #
    #     # 'dro_num_steps=16': f"../../results/exp_lp_dro_ns/success_ref/results/ppo/learning_progress/dro_num_steps_16",
    #     # 'dro_num_steps=32': f"../../results/exp_lp_dro_ns/success_ref/results/ppo/learning_progress/dro_num_steps_32",
    #     # 'dro_num_steps=64': f"../../results/exp_lp_dro_ns/success_ref/results/ppo/learning_progress/dro_num_steps_64",
    #     # 'dro_num_steps=128': f"../../results/exp_lp_dro_ns/success_ref/results/ppo/learning_progress/dro_num_steps_128",
    #     # 'dro_num_steps=256': f"../../results/exp_lp_dro_ns/success_ref/results/ppo/learning_progress/dro_num_steps_256",
    #     # 'dro_num_steps=512': f"../../results/exp_lp_dro_ns/success_ref/results/ppo/learning_progress/dro_num_steps_512",
    #     # 'dro_num_steps=1024': f"../../results/exp_lp_dro_ns/success_ref/results/ppo/learning_progress/dro_num_steps_1024",
    #
    #     # 'dro_step_size=0.01': f"../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.01",
    #     # 'dro_step_size=0.05': f"../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.05",
    #     # 'dro_step_size=0.1': f"../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.1",
    #     # 'dro_step_size=0.2': f"../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.2",
    #     # 'dro_step_size=0.4': f"../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.4",
    #     # 'dro_step_size=0.6': f"../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.6",
    #     # 'dro_step_size=0.8': f"../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.8",
    #     # 'dro_step_size=1.0': f"../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_1.0",
    #
    #     # 'dro_eta=1': f"../../results/exp_1/success_ref/results/ppo/dro/dro_eta_1/dro_step_size_0.2",
    #     # 'dro_eta=2': f"../../results/exp_1/success_ref/results/ppo/dro/dro_eta_2/dro_step_size_0.2",
    #     # 'dro_eta=4': f"../../results/exp_1/success_ref/results/ppo/dro/dro_eta_4/dro_step_size_0.2",
    #     # 'dro_eta=8': f"../../results/exp_1/success_ref/results/ppo/dro/dro_eta_8/dro_step_size_0.2",
    #     # 'dro_eta=16': f"../../results/exp_1/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.2",
    #     # 'dro_eta=32': f"../../results/exp_1/success_ref/results/ppo/dro/dro_eta_32/dro_step_size_0.2",
    #     # 'dro_eta=64': f"../../results/exp_1/success_ref/results/ppo/dro/dro_eta_64/dro_step_size_0.2",
    # }
    color_palette = iter(seaborn.color_palette('colorblind', n_colors=10))

    for key, results_dir in path_dict.items():
            x, y = get_data(results_dir, x_name='update',y_name=f'success_rate')
            T = (int)(len(x)) # reduce this value if you want to truncate the data
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
    save_name = f'{save_title}_success_rate_avg.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=200)
    print(f'save to {save_dir}/{save_name}')

    plt.show()


