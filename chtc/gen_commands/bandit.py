import os

if __name__ == "__main__":

    os.makedirs('../commands', exist_ok=True)
    f = open(f"../commands/example.txt", "w")

    for lr in [1e-2, 1e-3]:
        for ns in [64, 128, 256, 512]:
            python_command = (
                f'python ppo_discrete_action_dro_resampling.py '
                f' --output_subdir standard/lr_{lr}/ns_{ns} '
                f' --total_timesteps {ns * 10000}'
                f' --eval_freq 50'
                f' --num_steps {ns}  '
                f' --learning_rate {lr}'
                # f' --dro --dro_learning_rate {1} --dro_num_steps {ns // 4} --dro_eps 0.01'
            )

            mem = 0.5
            disk = 0.5
            command = f"{mem},{disk},{python_command}"
            print(command)

            f.write(command + "\n")

    for lr in [1e-2, 1e-3]:
        for ns in [128, 256, 512]:
            python_command = (
                f'python ppo_discrete_action_dro_resampling.py '
                f' --output_subdir dro/lr_{lr}/ns_{ns} '
                f' --total_timesteps {ns * 10000}'
                f' --eval_freq 50'
                f' --num_steps {ns}  '
                f' --learning_rate {lr}'
                f' --dro --dro_learning_rate {1} --dro_num_steps {ns // 4} --dro_eps 0.01'
            )

            mem = 0.5
            disk = 0.5
            command = f"{mem},{disk},{python_command}"
            print(command)

            f.write(command + "\n")
