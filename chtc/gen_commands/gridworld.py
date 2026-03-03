import os

# algo_list = ['dro', 'learning_progress', 'uniform', 'easy_first', 'hard_first', 'dro_reweight', ]
algo_list = ['learning_progress']
# algo_list = ['dro']
dro_eta_list = [1, 2, 4, 8, 16, 32, 64]
# dro_eta_list = [16]
dro_step_size_list = [0.2]
# dro_step_size_list = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
num_steps_list = [256]
# num_steps_list = [256, 512, 1024, 2048]
dro_num_steps_list = [256]
# dro_num_steps_list = [16, 32, 64, 128, 256, 512, 1024]

if __name__ == "__main__":

    lr = 3e-3
    for ns in num_steps_list:
        for algo in algo_list:
            for dro_eta in dro_eta_list:
                for dro_step_size in dro_step_size_list:
                    for dro_num_steps in dro_num_steps_list:
                        python_command = (
                            f'python ppo_discrete_action_dro.py '
                            # f' --output_subdir {algo}/lr_{lr}/ns_{ns}'
                            f' --output_subdir {algo}/dro_eta_{dro_step_size}'
                            f' --total_timesteps {int(ns/256 * 1000e3)}'
                            f' --num_evals 50'
                            f' --num_steps {ns}  '
                            f' --learning_rate {lr}'
                            f' --task_sampling_algo {algo}'
                            f' --dro_num_steps {dro_num_steps}'
                            f' --dro_eta {dro_eta}'
                            f' --dro_step_size {dro_step_size}'
                        )

                        mem = 0.4
                        disk = 2
                        command = f"{mem},{disk},{python_command}"
                        print(command)

