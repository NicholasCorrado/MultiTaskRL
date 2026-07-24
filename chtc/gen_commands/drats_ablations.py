"""Generate CHTC command files for the DRATS paper ablations (Appendix F).

Reproduces the three ablations from the paper for four algorithms:

  F.2 Temperature (eta):        eta in {1, 2, 4, 8, 16, 32, 64}, step_size fixed.
  F.2 Mirror-ascent step size:  step_size in {0.01, 0.05, 0.1, 0.2, 0.4, 0.6,
                                0.8, 1.0}, eta fixed. In the code, `dro_step_size`
                                is the interpolation coefficient c toward the
                                best-response q* = softmax(eta * gap); c = 1 is the
                                full best response (paper: alpha = c * eta).
  F.1 Sampling vs reweighting:  {uniform, adaptive-sampling (DRATS),
                                objective-reweighting (Uniform + DRATS-Reweight)}.

Targets:
  ddpg        -> ddpg_continuous_actions_drats.py   (off-policy)
  td3         -> td3_continuous_actions_drats.py    (off-policy)
  ppo_sepnn   -> ppo_continuous_action_drats_sepnn.py (separate net per task)
  gridworld   -> ppo_discrete_action_dro.py         (paper Fig 14/15 reproduction)

Run:
    cd chtc/gen_commands
    python drats_ablations.py

This writes one command file per (target, ablation) to ../commands/ablations/,
e.g. ../commands/ablations/ddpg_ablate_eta.txt. Each line is
"<mem>,<disk>,<command>"; job.sh appends "--run_id <step> --seed <step>", so the
number of seeds is chosen at submit time via ./submit.sh's second argument.

Submit (from chtc/, see the block printed at the end):
    ./submit.sh ablations/ddpg_ablate_eta 20
"""

import os


# --------------------------------------------------------------------------- #
# Paper ablation sweeps (Appendix F)
# --------------------------------------------------------------------------- #
ETA_LIST = [1, 2, 4, 8, 16, 32, 64]                       # F.2, Fig. 15a
STEP_SIZE_LIST = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]  # F.2, Fig. 15b

# Base values held fixed while the other hyperparameter is swept. These match
# the code defaults and the existing gridworld ablation scripts. The paper's
# most data-efficient gridworld setting is eta=8; edit BASE_ETA to 8 to center
# the step-size sweep there instead.
BASE_ETA = 16
BASE_STEP_SIZE = 0.2


# --------------------------------------------------------------------------- #
# Environment for the continuous-control targets.
#
# The new continuous scripts default to the gymnasium-robotics PointMaze suite.
# NOTE: PointMaze_*-v3 requires the `gymnasium-robotics` package, which chtc/job.sh
# does NOT currently install -- add this line to job.sh before running these:
#     python -m pip install gymnasium-robotics --target local
# To use a different multi-task continuous benchmark, edit CONT_ENV_IDS.
# --------------------------------------------------------------------------- #
CONT_ENV_IDS = "PointMaze_UMaze-v3 PointMaze_Medium-v3 PointMaze_Large-v3"


# --------------------------------------------------------------------------- #
# Per-target configuration.
#
#   script      : entry-point run on the compute node
#   env_ids     : value for --env_ids (None -> use the script's own default)
#   mem, disk   : per-job request in GB (first two columns of each command line)
#   algos       : task_sampling_algo names for each reweighting arm. The discrete
#                 script uses "dro"/"dro_reweight"; the continuous scripts use
#                 "drats"/"drats_reweight".
#   base        : shared flags (budget, eval cadence, optimizer, dro_num_steps)
# --------------------------------------------------------------------------- #
TARGETS = {
    "ddpg": {
        "script": "ddpg_continuous_actions_drats.py",
        "env_ids": CONT_ENV_IDS,
        "mem": 2,
        "disk": 4,
        "algos": {"uniform": "uniform", "sample": "drats", "reweight": "drats_reweight"},
        "base": (
            "--total_timesteps 500000 "
            "--num_evals 50 "
            "--learning_starts 1000 "
            "--batch_size 64 "
            "--learning_rate 0.0001 "
            "--dro_num_steps 1600"
        ),
    },
    "td3": {
        "script": "td3_continuous_actions_drats.py",
        "env_ids": CONT_ENV_IDS,
        "mem": 2,
        "disk": 4,
        "algos": {"uniform": "uniform", "sample": "drats", "reweight": "drats_reweight"},
        "base": (
            "--total_timesteps 500000 "
            "--num_evals 50 "
            "--learning_starts 1000 "
            "--batch_size 64 "
            "--learning_rate 0.0001 "
            "--dro_num_steps 1600"
        ),
    },
    "ppo_sepnn": {
        "script": "ppo_continuous_action_drats_sepnn.py",
        "env_ids": CONT_ENV_IDS,
        "mem": 5,
        "disk": 4,
        "algos": {"uniform": "uniform", "sample": "drats", "reweight": "drats_reweight"},
        "base": (
            "--total_timesteps 10000000 "
            "--num_evals 40 "
            "--num_steps 2048 "
            "--dro_num_steps 2048 "
            "--learning_rate 0.0003"
        ),
    },
    "gridworld": {
        "script": "ppo_discrete_action_dro.py",
        "env_ids": None,  # defaults to the 4-task GridWorld0..3
        "mem": 0.4,
        "disk": 2,
        "algos": {"uniform": "uniform", "sample": "dro", "reweight": "dro_reweight"},
        "base": (
            "--total_timesteps 1000000 "
            "--num_evals 50 "
            "--num_steps 256 "
            "--dro_num_steps 256 "
            "--learning_rate 0.003"
        ),
    },
}


def _prefix(cfg, algo, eta, step_size, subdir):
    """Assemble one full command line (without the leading mem,disk columns)."""
    parts = [
        f"python {cfg['script']}",
        f"--output_subdir {subdir}",
        cfg["base"],
        f"--task_sampling_algo {algo}",
        f"--dro_eta {eta}",
        f"--dro_step_size {step_size}",
    ]
    if cfg["env_ids"] is not None:
        parts.append(f"--env_ids {cfg['env_ids']}")
    return " ".join(parts)


def build_eta(cfg):
    lines = []
    for eta in ETA_LIST:
        subdir = f"ablate_eta/eta_{eta}"
        cmd = _prefix(cfg, cfg["algos"]["sample"], eta, BASE_STEP_SIZE, subdir)
        lines.append(cmd)
    return lines


def build_step_size(cfg):
    lines = []
    for step_size in STEP_SIZE_LIST:
        subdir = f"ablate_step_size/eta_{BASE_ETA}/step_size_{step_size}"
        cmd = _prefix(cfg, cfg["algos"]["sample"], BASE_ETA, step_size, subdir)
        lines.append(cmd)
    return lines


def build_reweight(cfg):
    lines = []
    for arm in ("uniform", "sample", "reweight"):
        algo = cfg["algos"][arm]
        subdir = f"ablate_reweight/{arm}"
        cmd = _prefix(cfg, algo, BASE_ETA, BASE_STEP_SIZE, subdir)
        lines.append(cmd)
    return lines


ABLATIONS = {
    "eta": build_eta,
    "step_size": build_step_size,
    "reweight": build_reweight,
}


if __name__ == "__main__":
    out_dir = "../commands/ablations"
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for target, cfg in TARGETS.items():
        for ablation, builder in ABLATIONS.items():
            lines = builder(cfg)
            name = f"{target}_ablate_{ablation}"
            path = os.path.join(out_dir, f"{name}.txt")
            with open(path, "w", encoding="utf-8") as f:
                for cmd in lines:
                    f.write(f"{cfg['mem']},{cfg['disk']},{cmd}\n")
            written.append((f"ablations/{name}", len(lines)))
            print(f"[{len(lines):2d} cmds] {path}")

    print("\nSubmit each ablation with (second arg = number of seeds):")
    for name, n in written:
        seeds = 50 if name.startswith("ablations/gridworld") else 20
        print(f"  ./submit.sh {name} {seeds}")
