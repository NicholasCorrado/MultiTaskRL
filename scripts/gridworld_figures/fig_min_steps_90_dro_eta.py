import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from plotting.utils import get_data
from scripts.gridworld_figures.common import first_reach_threshold_updates
from scripts.gridworld_figures.common import plot_with_ci
from scripts.gridworld_figures.common import save_figure


PATHS_BY_VALUE = {
    0: "../../results/exp_1/dro_eta=1/results/ppo/dro/lr_0.003/ns_256",
    1: "../../results/exp_1/dro_eta=2/results/ppo/dro/lr_0.003/ns_256",
    2: "../../results/exp_1/dro_eta=4/results/ppo/dro/lr_0.003/ns_256",
    3: "../../results/exp_1/dro_eta=8/results/ppo/dro/lr_0.003/ns_256",
    4: "../../results/exp_1/dro_eta=16/results/ppo/dro/lr_0.003/ns_256",
    5: "../../results/exp_1/dro_eta=32/results/ppo/dro/lr_0.003/ns_256",
    6: "../../results/exp_1/dro_eta=64/results/ppo/dro/lr_0.003/ns_256",
}


if __name__ == "__main__":
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)

    per_val_first_updates = []
    valid_vals = []
    for val, results_dir in PATHS_BY_VALUE.items():
        x, y = get_data(results_dir, x_name="timestep", y_name="success_rate")
        if x is None or y is None:
            continue
        updates = first_reach_threshold_updates(x, y, threshold=0.9)
        if updates.size == 0:
            continue
        valid_vals.append(val)
        per_val_first_updates.append(updates)

    common_runs = min(arr.size for arr in per_val_first_updates)
    first_update_matrix = np.stack([arr[:common_runs] for arr in per_val_first_updates], axis=1)

    method_name = "dro/dro_eta"
    x_dict = {method_name: np.array(valid_vals, dtype=float)}
    y_dict = {method_name: first_update_matrix}
    linestyle_dict = {method_name: "-"}
    color_dict = {method_name: seaborn.color_palette("colorblind", n_colors=1)[0]}

    plot_with_ci(
        ax,
        x_dict,
        y_dict,
        linestyle_dict,
        color_dict,
        xlabel="dro_eta (log2 scale index)",
        ylabel="Timestep at first 90% success",
    )
    plt.xticks(valid_vals)
    plt.grid(alpha=0.25)
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize("large")
    plt.tight_layout()
    save_figure(fig, "min_steps_to_90_dro_eta.png")
    plt.show()
