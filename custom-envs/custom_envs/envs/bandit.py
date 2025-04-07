from typing import Optional

import gymnasium as gym
import numpy as np
from sympy.codegen.ast import float32


class Bandit(gym.Env):
    def __init__(self, n=10, task_id=0):
        # np.random.seed(0)

        self.n = n
        self.action_space = gym.spaces.Discrete(self.n)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0,  0, 0, 0,0], np.float32), high=np.array([1, 1, 1, 1,  1, 1], np.float32),)
        self.task_id = [0, 0, 0, 0,0,]
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
        info = {'is_success': a == np.argmax(self.means)}
        return np.array(self.task_id + [0]), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        return np.array(self.task_id + [0]), {}


class BanditEasy(Bandit):
    def __init__(self, n=64, task_id=0):
        super().__init__(n=n)
        self.task_id[0] = 1

        self.means = np.random.uniform(0, 0.5, self.n)
        self.stds = np.random.uniform(0, 0.1, self.n)

        self.means[0] = 1
        self.stds[0] = 0.1


class BanditHard(Bandit):
    def __init__(self, n=64, opt_std=0.5, subopt_std=0.5):
        super().__init__(n=n)
        self.task_id[1] = 1


        self.means = np.random.uniform(0, 0.9, self.n)
        self.stds = np.random.uniform(0, 0.1, self.n)

        self.means[-1] = 1
        self.stds[-1] = 0.1


class Bandit1(Bandit):
    def __init__(self, n=64, opt_std=0.5, subopt_std=0.5):
        super().__init__(n=n)
        self.task_id[0] = 1


        self.means = np.random.uniform(0, 0.2, self.n)
        self.stds = np.random.uniform(0, 0.1, self.n)

        self.means[:5] = 0
        self.means[0] = 1
        self.stds[0] = 0.01

class Bandit2(Bandit):
    def __init__(self, n=64, opt_std=0.5, subopt_std=0.5):
        super().__init__(n=n)
        self.task_id[1] = 1


        self.means = np.random.uniform(0, 0.4, self.n)
        self.stds = np.random.uniform(0, 0.1, self.n)

        self.means[:5] = 0
        self.means[1] = 1
        self.stds[1] = 0.01

class Bandit3(Bandit):
    def __init__(self, n=64, opt_std=0.5, subopt_std=0.5):
        super().__init__(n=n)
        self.task_id[2] = 1


        self.means = np.random.uniform(0, 0.6, self.n)
        self.stds = np.random.uniform(0, 0.1, self.n)

        self.means[:5] = 0
        self.means[2] = 1
        self.stds[2] = 0.01

class Bandit4(Bandit):
    def __init__(self, n=64, opt_std=0.5, subopt_std=0.5):
        super().__init__(n=n)
        self.task_id[3] = 1


        self.means = np.random.uniform(0, 0.8, self.n)
        self.stds = np.random.uniform(0, 0.1, self.n)

        self.means[:5] = 0
        self.means[3] = 1
        self.stds[3] = 0.01

class Bandit5(Bandit):
    def __init__(self, n=64, opt_std=0.5, subopt_std=0.5):
        super().__init__(n=n)
        self.task_id[4] = 1


        self.means = np.random.uniform(0, 0.9, self.n)
        self.stds = np.random.uniform(0, 0.1, self.n)

        self.means[:5] = 0
        self.means[4] = 1
        self.stds[4] = 0.01