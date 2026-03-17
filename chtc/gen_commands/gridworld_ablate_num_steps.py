import os


LR = 3e-3
NUM_STEPS_LIST = [256, 512, 1024, 2048]
ALGO_LIST = ["dro", "learning_progress", "uniform"]


def build_command(algo, num_steps):
    return (
        "python ppo_discrete_action_dro.py "
        f"--output_subdir {algo}/num_steps_{num_steps} "
        f"--total_timesteps {int(num_steps / 256 * 1000e3)} "
        "--num_evals 50 "
        f"--num_steps {num_steps} "
        f"--learning_rate {LR} "
        f"--task_sampling_algo {algo} "
        "--dro_num_steps 256 "
        "--dro_eta 16 "
        "--dro_step_size 0.2"
    )


if __name__ == "__main__":
    os.makedirs("../commands", exist_ok=True)
    output_path = "../commands/gridworld_ablate_num_steps.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for algo in ALGO_LIST:
            for num_steps in NUM_STEPS_LIST:
                command = f"0.4,2,{build_command(algo, num_steps)}"
                print(command)
                f.write(command + "\n")

    print(f"Saved commands to: {output_path}")
