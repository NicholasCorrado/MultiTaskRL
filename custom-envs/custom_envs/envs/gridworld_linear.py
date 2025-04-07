from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


class GridWorldLinearEnv(gym.Env):
    def __init__(self, shape=(1,50), rewards=(-0.01, 1)):
        super().__init__()

        self.shape = np.array(shape)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(np.product(self.shape),))
        self.taskid_easy = 0
        self.taskid_hard = 0

        self.nrows, self.ncols = self.shape
        self.rowcol = (0, 10)
        self.init_rowcol = (0, 10)

        self.rewards = rewards[0] * np.ones(shape=self.shape)
        # self.rewards[:3, :3] = 0.1
        self.opt_reward = rewards[1]

        self.terminals = np.zeros(shape=self.shape, dtype=bool)
        self.terminals[self.rewards > 0] = True
        # self.terminals[self.rewards > 0.02] = True

        print(self.rewards)
        print(self.terminals)


    def _rowcol_to_obs(self, rowcol):
        idx = int(rowcol[0] * self.shape[0] + rowcol[1])
        state = np.zeros(self.observation_space.shape[-1])
        state[idx] = 1
        return state

    def step(self, a):
        # up
        if a == 0:
            self.rowcol[1] -= 1
        # down
        elif a == 1:
            self.rowcol[1] += 1

        self.rowcol = np.clip(self.rowcol, a_min=np.zeros(2), a_max=self.shape-1).astype(int)

        state = self._rowcol_to_obs(self.rowcol)
        reward = self.rewards[self.rowcol[0], self.rowcol[1]]
        terminated = self.terminals[self.rowcol[0], self.rowcol[1]]
        truncated = False
        info = {'is_success': reward == self.opt_reward}

        return np.array([self.taskid_easy] + [self.taskid_hard] + state), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.rowcol = self.init_rowcol.copy()
        state = self._rowcol_to_obs(self.rowcol)

        return state, {}


class GridWorldLinearEasy(GridWorldLinearEnv):
    def __init__(self, shape=(1,50), rewards=(-0.01, 1)):
        super().__init__(shape, rewards)

        self.taskid_easy = 1
        self.taskid_hard = 0

        self.rewards[0, 0] = rewards[1]


class GridWorldLinearHard(GridWorldLinearEnv):
    def __init__(self, shape=(1,50), rewards=(-0.01, 1)):
        super().__init__(shape, rewards)

        self.taskid_easy = 0
        self.taskid_hard = 1

        self.rewards[self.nrows-1, self.ncols-1] = rewards[1]