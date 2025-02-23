from collections import defaultdict

import numpy as np
import torch

def simulate(env, actor, eval_episodes, eval_steps=np.inf):
    logs = defaultdict(list)
    step = 0
    num_env = 0
    for episode_i in range(eval_episodes):
        logs_episode = defaultdict(list)

        obs, _ = env.reset()
        done = False
        Done = False

        while not Done:

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                actions = actor.get_action(torch.Tensor(obs).to('cpu'))
                actions = actions.cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            done = np.logical_or(terminateds, truncateds)
            Done = done.all()

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            logs_episode['rewards'].append(rewards)

            step += 1

            num_env = len(rewards)

            if step >= eval_steps:
                break
        if step >= eval_steps:
            break

        logs['returns'].append(logs_episode['rewards'])
        logs['returns_avg'].append(np.mean(logs_episode['rewards']))
        try:
            logs['successes'].append(infos['is_success'])
        except:
            logs['successes'].append(False)

    returns = np.mean(logs['returns'], axis = 0)
    return_avg = np.mean(logs['returns_avg'])
    return_std = np.std(logs['returns'])
    success_avg = np.mean(logs['successes'])
    success_std = np.std(logs['successes'])
    return returns, return_avg, return_std, success_avg, success_std
    # return np.array(eval_returns), np.array(eval_obs), np.array(eval_actions), np.array(eval_rewards)
