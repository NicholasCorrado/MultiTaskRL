from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

def get_obs_info(obs, infos, task_id):
    next_obs = [obs['observation'][i] for i in range(4)] + [obs['achieved_goal'][i] for i in range(2)] + [obs['desired_goal'][i] for i in range(2)] + task_id
    infos = {'is_success': infos['success'], 'success': infos['success']}

    next_obs = np.array(next_obs)

    return next_obs, infos

def get_obs(obs, task_id):
    next_obs = [obs['observation'][i] for i in range(4)] + [obs['achieved_goal'][i] for i in range(2)] + [obs['desired_goal'][i] for i in range(2)] + task_id

    next_obs = np.array(next_obs)

    return next_obs

class PointMazeEnv1(gym.Env):
    def __init__(self):

        self.env = gym.make('PointMaze_Open-v3')
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(np.array([-np.inf for _ in range(8)] + [0 for _ in range(3)]),
                                                np.array([np.inf for _ in range(8)] + [1 for _ in range(3)]),
                                                (11,),
                                                np.float64)
        self.task_id = [1, 0, 0]

        super().__init__()

    def step(self, a):

        obs, reward, terminations, truncations, infos = self.env.step(a)
        next_obs, infos = get_obs_info(obs, infos, self.task_id)

        return next_obs, reward, terminations, truncations, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        obs, _ = self.env.reset()
        next_obs = get_obs(obs, self.task_id)

        return next_obs, {}


class PointMazeEnv2(gym.Env):
    def __init__(self):

        self.env = gym.make('PointMaze_Medium-v3')
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(np.array([-np.inf for _ in range(8)] + [0 for _ in range(3)]),
                                                np.array([np.inf for _ in range(8)] + [1 for _ in range(3)]),
                                                (11,),
                                                np.float64)
        self.task_id = [0, 1, 0]

        super().__init__()

    def step(self, a):

        obs, reward, terminations, truncations, infos = self.env.step(a)
        next_obs, infos = get_obs_info(obs, infos, self.task_id)

        return next_obs, reward, terminations, truncations, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        obs, _ = self.env.reset()
        next_obs = get_obs(obs, self.task_id)

        return next_obs, {}


class PointMazeEnv3(gym.Env):
    def __init__(self):

        self.env = gym.make('PointMaze_Large-v3')
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(np.array([-np.inf for _ in range(8)] + [0 for _ in range(3)]),
                                                np.array([np.inf for _ in range(8)] + [1 for _ in range(3)]),
                                                (11,),
                                                np.float64)
        self.task_id = [0, 0, 1]

        super().__init__()

    def step(self, a):

        obs, reward, terminations, truncations, infos = self.env.step(a)
        next_obs, infos = get_obs_info(obs, infos, self.task_id)

        return next_obs, reward, terminations, truncations, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        obs, _ = self.env.reset()
        next_obs = get_obs(obs, self.task_id)

        return next_obs, {}
