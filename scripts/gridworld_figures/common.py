import os

import numpy as np
import seaborn
from rliable import library as rly
from rliable import metrics

from plotting.utils import get_data
from plotting.utils import plot_sample_efficiency_curve


def build_styles(keys):
    palette = seaborn.color_palette("colorblind", n_colors=max(10, len(keys)))
    color_dict = dict(zip(keys, palette))
    linestyle_dict = {k: "-" for k in keys}
    return linestyle_dict, color_dict


def collect_curves(path_dict, y_name, x_name="timestep"):
    x_dict, y_dict = {}, {}
    for key, results_dir in path_dict.items():
        x, y = get_data(results_dir, x_name=x_name, y_name=y_name)
        if x is None or y is None:
            continue
        x_dict[key] = x
        y_dict[key] = y
    return x_dict, y_dict


def plot_with_ci(
    ax,
    x_dict,
    y_dict,
    linestyle_dict,
    color_dict,
    xlabel,
    ylabel,
    reps=1000,
):
    results_dict = {algorithm: score for algorithm, score in y_dict.items()}
    aggr_func = lambda scores: np.array(
        [metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])]
    )
    scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=reps)

    plot_sample_efficiency_curve(
        frames=x_dict,
        point_estimates=scores,
        interval_estimates=cis,
        ax=ax,
        algorithms=None,
        xlabel=xlabel,
        ylabel=ylabel,
        labelsize="large",
        ticklabelsize="large",
        linestyles=linestyle_dict,
        colors=color_dict,
        marker="",
    )


def first_reach_threshold_updates(x, y, threshold=0.9):
    first_updates = []
    for run_success in y:
        reached = np.where(run_success >= threshold)[0]
        if reached.size > 0:
            first_updates.append(x[reached[0]])
    return np.array(first_updates, dtype=float)


def save_figure(fig, save_name):
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{save_name}", dpi=200)
    print(f"Saved figure to figures/{save_name}")
