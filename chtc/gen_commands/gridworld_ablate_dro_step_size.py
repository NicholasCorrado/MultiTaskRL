import os


LR = 3e-3
NUM_STEPS = 256
DRO_ETA = 16
DRO_NUM_STEPS = 256
DRO_STEP_SIZE_LIST = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def build_command(dro_step_size):
    return (
        "python ppo_discrete_action_dro.py "
        f"--output_subdir dro/dro_eta_{DRO_ETA}/dro_step_size_{dro_step_size} "
        f"--total_timesteps {int(NUM_STEPS / 256 * 1000e3)} "
        "--num_evals 50 "
        f"--num_steps {NUM_STEPS} "
        f"--learning_rate {LR} "
        "--task_sampling_algo dro "
        f"--dro_num_steps {DRO_NUM_STEPS} "
        f"--dro_eta {DRO_ETA} "
        f"--dro_step_size {dro_step_size}"
    )


if __name__ == "__main__":
    os.makedirs("../commands", exist_ok=True)
    output_path = "../commands/gridworld_ablate_dro_step_size.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for dro_step_size in DRO_STEP_SIZE_LIST:
            command = f"0.4,2,{build_command(dro_step_size)}"
            print(command)
            f.write(command + "\n")

    print(f"Saved commands to: {output_path}")
