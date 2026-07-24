import os


LR = 3e-3
NUM_STEPS = 256
DRO_ETA = 16
DRO_STEP_SIZE = 0.2
DRO_NUM_STEPS_LIST = [16, 32, 64, 128, 256, 512, 1024]


def build_command(dro_num_steps):
    return (
        "python ppo_discrete_action_dro.py "
        f"--output_subdir dro/dro_eta_{DRO_ETA}/dro_num_steps_{dro_num_steps} "
        f"--total_timesteps {int(NUM_STEPS / 256 * 1000e3)} "
        "--num_evals 50 "
        f"--num_steps {NUM_STEPS} "
        f"--learning_rate {LR} "
        "--task_sampling_algo dro "
        f"--dro_num_steps {dro_num_steps} "
        f"--dro_eta {DRO_ETA} "
        f"--dro_step_size {DRO_STEP_SIZE}"
    )


if __name__ == "__main__":
    os.makedirs("../commands", exist_ok=True)
    output_path = "../commands/gridworld_ablate_dro_num_steps.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for dro_num_steps in DRO_NUM_STEPS_LIST:
            command = f"0.4,2,{build_command(dro_num_steps)}"
            print(command)
            f.write(command + "\n")

    print(f"Saved commands to: {output_path}")
