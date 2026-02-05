import os

if __name__ == "__main__":

    dro_eta = 8
    dro_step_size = 0
    dro_eps = 0.05

    for lr in [3e-3]:
        for ns in [512, 1024]:
            for dro in [1]:
                python_command = (
                    f'python ppo_discrete_action_dro.py '
                    f' --output_subdir dro_{dro}/lr_{lr}/ns_{ns} '
                    f' --total_timesteps {int(ns/128 * 500e3)}'
                    f' --num_evals 50'
                    f' --num_steps {ns}  '
                    f' --learning_rate {lr}'
                    f' --dro {dro}'
                    f' --dro_num_steps {ns}'
                    f' --dro_eta {dro_eta}'
                    f' --dro_step_size {dro_step_size}'
                    f' --dro_eps {dro_eps}'
                    # f' --dro --dro_learning_rate {1} --dro_num_steps {ns // 4} --dro_eps 0.01'
                )

                mem = 0.5
                disk = 2.5
                command = f"{mem},{disk},{python_command}"
                print(command)

