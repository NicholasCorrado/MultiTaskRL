import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from common import build_styles
from common import collect_curves
from common import plot_with_ci
from common import save_figure


PATH_DICT = {
    "eta=1.0": "results/exp_1_1/success_ref/results/ppo",
    "eta=2.0": "results/exp_1_2/success_ref/results/ppo",
    "eta=4.0": "results/exp_1_3/success_ref/results/ppo",
    "eta=8.0": "results/exp_1_4/success_ref/results/ppo",
    "eta=16.0": "results/exp_1_5/success_ref/results/ppo",
    "eta=32.0": "results/exp_1_6/success_ref/results/ppo",
    "eta=64.0": "results/exp_1_7/success_ref/results/ppo",
}


if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 4))
    linestyle_dict, color_dict = build_styles(PATH_DICT.keys())

    for task_i in range(4):
        ax = plt.subplot(1, 4, task_i + 1)
        x_dict, y_dict = collect_curves(PATH_DICT, y_name=f"task_probs_{task_i}")
        plot_with_ci(
            ax,
            x_dict,
            y_dict,
            linestyle_dict,
            color_dict,
            xlabel="Timestep",
            ylabel="Probability",
            reps=100,
        )
        ax.set_title(f"Task {task_i} sampling prob", fontsize="large")
        ax.set_ylim(0, 1.05)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize("large")
        if task_i == 0:
            ax.legend(fontsize="large")

    plt.tight_layout()
    save_figure(fig, "task_probs_dro_eta.png")
    plt.show()
