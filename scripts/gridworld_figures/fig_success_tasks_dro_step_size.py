import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from scripts.gridworld_figures.common import build_styles
from scripts.gridworld_figures.common import collect_curves
from scripts.gridworld_figures.common import plot_with_ci
from scripts.gridworld_figures.common import save_figure


PATH_DICT = {
    "dro_step_size=0.01": "../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.01",
    "dro_step_size=0.05": "../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.05",
    "dro_step_size=0.1": "../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.1",
    "dro_step_size=0.2": "../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.2",
    "dro_step_size=0.4": "../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.4",
    "dro_step_size=0.6": "../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.6",
    "dro_step_size=0.8": "../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_0.8",
    "dro_step_size=1.0": "../../results/exp_2/success_ref/results/ppo/dro/dro_eta_16/dro_step_size_1.0",
}


if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 4))
    linestyle_dict, color_dict = build_styles(PATH_DICT.keys())

    for task_i in range(4):
        ax = plt.subplot(1, 4, task_i + 1)
        x_dict, y_dict = collect_curves(PATH_DICT, y_name=f"success_rate_{task_i}")
        plot_with_ci(
            ax,
            x_dict,
            y_dict,
            linestyle_dict,
            color_dict,
            xlabel="Timestep",
            ylabel="Success rate",
            reps=100,
        )
        ax.set_title(f"Task {task_i}", fontsize="large")
        ax.set_ylim(0, 1.05)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize("large")
        if task_i == 0:
            ax.legend(fontsize="large")

    plt.tight_layout()
    save_figure(fig, "success_tasks_dro_step_size.png")
    plt.show()
