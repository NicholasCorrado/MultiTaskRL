import os

if __name__ == "__main__":

    dro_eta = 16
    dro_step_size = 0.2
    dro_eps = 0.01
    for ns in [256]:
        for lr in [3e-3]:
            for algo in ['dro', 'learning_progress', 'uniform', 'easy_first', 'hard_first', 'dro_reweight', ]:
                dro_num_steps = ns
                python_command = (
                    f'python ppo_discrete_action_dro.py '
                    f' --output_subdir {algo}/lr_{lr}/ns_{ns}'
                    f' --total_timesteps {int(ns/256 * 1000e3)}'
                    f' --num_evals 50'
                    f' --num_steps {ns}  '
                    f' --learning_rate {lr}'
                    f' --task_sampling_algo {algo}'
                    f' --dro_num_steps {dro_num_steps}'
                    f' --dro_eta {dro_eta}'
                    f' --dro_step_size {dro_step_size}'
                    f' --dro_eps {dro_eps}'
                    f' --ent_coef 0.00'
                )

                mem = 0.4
                disk = 2
                command = f"{mem},{disk},{python_command}"
                print(command)

