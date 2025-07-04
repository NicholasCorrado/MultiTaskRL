from typing import Optional

import gymnasium as gym
import numpy as np
import torch

class Goal2DEnv(gym.Env):
    def __init__(self, delta=0.1, sparse=1, quadrant=False, center=False, fixed_goal=None):

        self.n = 2
        self.action_space = gym.spaces.Box(low=-1*np.ones(2), high=np.ones(2), shape=(self.n,))

        self.boundary = 1.05
        # self.observation_space = gym.spaces.Box(-self.boundary, +self.boundary, shape=(2 * self.n,), dtype="float64")  # chessboard size
        self.observation_space = gym.spaces.Box(np.array([-self.boundary, -self.boundary, -self.boundary, -self.boundary, 0, 0, 0, 0]),
                                                np.array([+self.boundary, +self.boundary, +self.boundary, +self.boundary, 1, 1, 1, 1]),
                                                shape=(2 * self.n + 4,),
                                                dtype="float64")

        self.step_num = 0
        self.delta = delta

        self.sparse = sparse
        self.d = 1
        self.x_norm = None
        self.quadrant = quadrant
        self.center = center
        self.fixed_goal = fixed_goal
        self.task_id = [0, 0, 0, 0]
        super().__init__()

    def _clip_position(self):
        # Note: clipping makes dynamics nonlinear
        self.x = np.clip(self.x, -self.boundary, +self.boundary)

    def step(self, a):

        self.step_num += 1
        self.x += a * self.delta
        self._clip_position()

        dist = np.linalg.norm(self.x - self.goal)
        terminated = dist < 0.1
        truncated = False

        if self.sparse:
            reward = +1.0 if terminated else -0.1
        else:
            reward = -dist

        info = {'is_success': terminated}
        self.obs = np.concatenate((self.x, self.goal))
        return self._get_obs(), reward, terminated, truncated, info

    def _sample_goal(self):
        if self.quadrant:
            goal = np.random.uniform(low=0, high=1, size=(self.n,))
        elif self.fixed_goal:
            goal = self.fixed_goal
        else:
            goal = np.random.uniform(low=-self.d, high=self.d, size=(self.n,))
        return goal

    def _get_obs(self):
        return np.concatenate([self.x, self.goal, self.task_id])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        self.step_num = 0

        self.x = np.random.uniform(-1, 1, size=(self.n,))
        self.goal = self._sample_goal()
        self.obs = np.concatenate((self.x, self.goal))
        return self._get_obs(), {}

class Goal2DQuadrantEnv(Goal2DEnv):
    def __init__(self, d=1, rbf_n=None, d_fourier=None, neural=False):
        super().__init__(delta=0.025, sparse=1, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, quadrant=True)


class Goal2D1Env(Goal2DEnv):
    def __init__(self):
        super().__init__(delta = 0.1, sparse=0, quadrant=False, center=False, fixed_goal=False)
        self.task_id = [1, 0, 0, 0]

class Goal2D2Env(Goal2DEnv):
    def __init__(self):
        super().__init__(delta = 0.04, sparse=0, quadrant=False, center=False, fixed_goal=False)
        self.task_id = [0, 1, 0, 0]

class Goal2D3Env(Goal2DEnv):
    def __init__(self):
        super().__init__(delta = 0.02, sparse=0, quadrant=False, center=False, fixed_goal=False)
        self.task_id = [0, 0, 1, 0]

class Goal2D4Env(Goal2DEnv):
    def __init__(self):
        super().__init__(delta = 0.01, sparse=0, quadrant=False, center=False, fixed_goal=False)
        self.task_id = [0, 0, 0, 1]

class Goal2DEasyEnv(Goal2DEnv):
    def __init__(self):
        super().__init__(delta = 0.1, sparse=0, quadrant=False, center=False, fixed_goal=False)
        self.task_id = [1, 0, 0, 0]

class Goal2DHardEnv(Goal2DEnv):
    def __init__(self):
        super().__init__(delta = 0.02, sparse=0, quadrant=False, center=False, fixed_goal=False)
        self.task_id = [0, 0, 0, 1]

    def step(self, a):

        a = np.array([a[1], a[0]])
        self.step_num += 1
        self.x += a * self.delta
        self._clip_position()

        dist = np.linalg.norm(self.x - self.goal)
        terminated = dist < 0.1
        truncated = False

        if self.sparse:
            reward = +1.0 if terminated else -0.1
        else:
            reward = -dist

        info = {'is_success': terminated}
        self.obs = np.concatenate((self.x, self.goal))
        return self._get_obs(), reward, terminated, truncated, info
