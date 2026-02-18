import os

if __name__ == "__main__":

    dro_eta = 8
    dro_step_size = 0.1
    dro_eps = 0.05
    # dro_num_steps = 256
    algo = 'dro_reweight'
    for algo in ['dro_reweight', 'dro', 'slope', 'uniform', 'easy-to-hard']:
        for lr in [1e-3, 5e-4]:
            for ns in [32,64,128]:
                dro_num_steps = ns
                python_command = (
                    f'python ppo_discrete_action_dro.py '
                    f' --output_subdir {algo}/lr_{lr}/ns_{ns}/dns_{dro_num_steps}'
                    f' --total_timesteps {int(ns/128 * 500e3)}'
                    f' --num_evals 50'
                    f' --num_steps {ns}  '
                    f' --learning_rate {lr}'
                    f' --task_sampling_algo {algo}'
                    f' --dro_num_steps {dro_num_steps}'
                    f' --dro_eta {dro_eta}'
                    f' --dro_step_size {dro_step_size}'
                    f' --dro_eps {dro_eps}'
                    # f' --dro --dro_learning_rate {1} --dro_num_steps {ns // 4} --dro_eps 0.01'
                )

                mem = 0.4
                disk = 2
                command = f"{mem},{disk},{python_command}"
                print(command)

