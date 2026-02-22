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

    seaborn.set_theme(style='whitegrid')

    # Define subplot grid
    n_rows = 3
    n_cols = 3
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    i = 1

    env_ids = ['Task_1.0', 'Task_2.0', 'Task_3.0', 'Task_4.0', 
               'Task_5.0', 'Task_6.0', 'Task_7.0', 'Task_8.0', 'Task_9.0']

    for env_id in env_ids:
        key = "PPO"
        results_dir = f"../results/{env_id}/ppo/"
        
        if not os.path.exists(results_dir):
            print(f"Task {env_id} does not have results!")
            continue

        ax = plt.subplot(n_rows, n_cols, i)
        ax.set_title(env_id)
        i += 1

        # Load data
        x, y = get_data(results_dir, x_name='timestep', y_name='return')

        if isinstance(y, list):  # Handling return_list
            for env_index, env_results in enumerate(y):
                if isinstance(env_results, (list, np.ndarray)):
                    ax.plot(x[:len(env_results)], env_results, label=f'Env {env_index}')
        else:
            ax.plot(x, y, label='Mean Return')

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Return")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()
