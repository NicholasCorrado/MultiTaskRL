import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from scripts.gridworld_figures.common import build_styles
from scripts.gridworld_figures.common import collect_curves
from scripts.gridworld_figures.common import plot_with_ci
from scripts.gridworld_figures.common import save_figure


PATH_DICT = {
    r"$\alpha$=0.01": "../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.01",
    r"$\alpha$=0.05": "../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.05",
    r"$\alpha$=0.1": "../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.1",
    r"$\alpha$=0.2": "../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.2",
    r"$\alpha$=0.4": "../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.4",
    r"$\alpha$=0.6": "../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.6",
    r"$\alpha$=0.8": "../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_0.8",
    r"$\alpha$=1.0": "../../results/exp_lp_dro_step_size/success_ref/results/ppo/learning_progress/dro_step_size_1.0",
}


if __name__ == "__main__":
    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(1, 1, 1)
    linestyle_dict, color_dict = build_styles(PATH_DICT.keys())
    x_dict, y_dict = collect_curves(PATH_DICT, y_name="success_rate")

    plot_with_ci(
        ax,
        x_dict,
        y_dict,
        linestyle_dict,
        color_dict,
        xlabel="Timestep",
        ylabel="Success Rate",
    )
    plt.axhline(y=1, color="k", linestyle="--")
    plt.ylim(0, 1.1)
    plt.legend(fontsize="large")
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize("large")
    plt.tight_layout()
    save_figure(fig, "success_avg_learning_progress_dro_step_size.png")
    plt.show()
