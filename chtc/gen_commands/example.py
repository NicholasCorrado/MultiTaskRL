import os

if __name__ == "__main__":

    os.makedirs('commands', exist_ok=True)
    f = open(f"commands/example.txt", "w")
    
    timesteps = 1000000

    for env_id in ['Swimmer-v5', 'Hopper-v5', 'HalfCheetah-v5']:
        for lr in [3e-4, 1e-3]:
            for num_steps in [1024, 2048, 4096, 8192]:
                python_command = (
                    f"python ppo_continuous_action.py --env_id {env_id}" 
                    f" --learning_rate {lr}"
                    f" --num_steps {num_steps}"
                    f" --output_subdir lr_{lr}/ns_{num_steps}"
                    f" --total_timesteps {timesteps}"
                )

            mem = 1
            disk = 10
            command = f"{mem},{disk},{python_command}"

            f.write(command + "\n")
