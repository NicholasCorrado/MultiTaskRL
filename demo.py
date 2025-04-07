import os

for env_id in ['Swimmer-v5', 'Hopper-v5', 'HalfCheetah-v5', 'Walker2d-v5', 'Ant-v5', 'Humanoid-v5']:
    for seed in range(3):
        command = (
            f"python ppo_continuous_action.py --run-id {seed} --seed {seed} "
            f"--env-id {env_id} --total-timesteps 20000 --eval-freq 1 "
        )
        os.system(command)
