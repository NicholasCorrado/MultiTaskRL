import os

if __name__ == "__main__":

    os.makedirs('../commands', exist_ok=True)
    f = open(f"../commands/example.txt", "w")

    for lr in [1e-2]:
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

            mem = 0.3
            disk = 0.3
            command = f"{mem},{disk},{python_command}"
            print(command)

            f.write(command + "\n")

    for lr in [1e-2]:
        for ns in [64, 128, 256, 512]:
            python_command = (
                f'python ppo_discrete_action_dro_resampling.py '
                f' --output_subdir dro/return_ref/lr_{lr}/ns_{ns} '
                f' --total_timesteps {ns * 10000}'
                f' --eval_freq 50'
                f' --num_steps {ns}  '
                f' --learning_rate {lr}'
                f' --dro --dro_learning_rate {1} --dro_num_steps {ns // 4} --dro_eps 0.01'
            )

            mem = 0.3
            disk = 0.3
            command = f"{mem},{disk},{python_command}"
            print(command)

            f.write(command + "\n")

    for lr in [1e-2]:
        for ns in [64, 128, 256, 512]:
            python_command = (
                f'python ppo_discrete_action_dro_resampling.py '
                f' --output_subdir dro/success_ref/lr_{lr}/ns_{ns} '
                f' --total_timesteps {ns * 10000}'
                f' --eval_freq 50'
                f' --num_steps {ns}  '
                f' --learning_rate {lr}'
                f' --dro --dro_success_ref --dro_learning_rate {1} --dro_num_steps {ns // 4} --dro_eps 0.01'
            )

            mem = 0.3
            disk = 0.3
            command = f"{mem},{disk},{python_command}"
            print(command)

            f.write(command + "\n")
