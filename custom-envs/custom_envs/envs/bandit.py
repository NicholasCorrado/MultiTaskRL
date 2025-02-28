from typing import Optional

import gymnasium as gym
import numpy as np
from sympy.codegen.ast import float32


class Bandit(gym.Env):
    def __init__(self, n=10, task_id=0):

        self.n = n
        self.action_space = gym.spaces.Discrete(self.n)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0], np.float32), high=np.array([np.inf, 1], np.float32),)
        self.task_id = task_id
        super().__init__()

        self.means = np.random.uniform(0, 0.9, self.n)
        self.stds = np.random.uniform(0, 1, self.n)
        # self.means = np.array([0.6, 0.7, 0.8, 0.9, 1])
        # self.stds[:] = 0.5
        self.means[-1] = 1
        self.stds[-1] = 0.1


    def step(self, a):

        reward = np.random.normal(self.means[a], self.stds[a])
        terminated = True
        truncated = False
        info = {}
        return np.array([self.task_id] + [1]), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        return np.array([self.task_id] + [1]), {}


class BanditEasy(Bandit):
    def __init__(self, n=100, reward=1, task_id=0):
        super().__init__(n=n)

        self.means = np.random.uniform(0, 0.7, self.n)
        self.stds = np.random.uniform(0, 1, self.n)
        self.task_id = task_id

        self.means[0] = 1
        self.stds[0] = 0.1


class BanditHard(Bandit):
    def __init__(self, n=100, reward=1, task_id=0):
        super().__init__(n=n)

        self.means = np.random.uniform(0, 0.9, self.n)
        self.stds = np.random.uniform(0, 1, self.n)

        self.means[-1] = 1
        self.stds[-1] = 1