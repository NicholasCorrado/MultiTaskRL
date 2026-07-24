import os


LR = 3e-3
NUM_STEPS = 256
TOTAL_TIMESTEPS = int(1000e3)
NUM_EVALS = 50
DRO_NUM_STEPS = 256
DRO_ETA = 16
DRO_STEP_SIZE = 0.2

ALGO_LIST = [
    "uniform",
    "learning_progress",
    "learning_progress_v",
]


def build_command(algo):
    return (
        "python ppo_discrete_action_dro.py "
        "--output_rootdir results/exp_baselines "
        f"--output_subdir {algo} "
        f"--total_timesteps {TOTAL_TIMESTEPS} "
        f"--num_evals {NUM_EVALS} "
        f"--num_steps {NUM_STEPS} "
        f"--learning_rate {LR} "
        f"--task_sampling_algo {algo} "
        f"--dro_num_steps {DRO_NUM_STEPS} "
        f"--dro_eta {DRO_ETA} "
        f"--dro_step_size {DRO_STEP_SIZE}"
    )


if __name__ == "__main__":
    os.makedirs("../commands", exist_ok=True)
    output_path = "../commands/gridworld_baselines.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for algo in ALGO_LIST:
            command = f"0.4,2,{build_command(algo)}"
            print(command)
            f.write(command + "\n")

    print(f"Saved commands to: {output_path}")
