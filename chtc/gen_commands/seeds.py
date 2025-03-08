import os

if __name__ == "__main__":

    os.makedirs('../commands', exist_ok=True)
    f = open(f"../commands/seeds_20.txt", "w")

    timesteps = 200000

    env_id = "BanditHard-v0"
    for seed in range(20):
        python_command = (
            f"python ppo_discrete_action.py --env_id {env_id}"
            f" --seed {seed}"
        )

        mem = 1
        disk = 10
        command = f"{python_command}"

        f.write(command + "\n")
