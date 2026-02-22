"""
Gridworld environment with exposed MDP information for exact computation.
"""

import gymnasium as gym
import numpy as np

R = 'r'
G = 'g'

gridworld_maps = [
    # [[R, 0, 0, 0, G],
    #  [0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0]],
    #
    # [[R, 0, 1, 0, G],
    #  [0, 0, 1, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0]],
    #
    # [[R, 0, 1, 0, G],
    #  [0, 0, 1, 0, 0],
    #  [0, 0, 1, 0, 0],
    #  [0, 0, 1, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0]],
    #
    # [[R, 0, 1, 0, G],
    #  [0, 0, 1, 0, 0],
    #  [0, 0, 1, 0, 0],
    #  [0, 0, 1, 0, 0],
    #  [0, 0, 1, 0, 0],
    #  [0, 0, 1, 0, 0],
    #  [0, 0, 0, 0, 0]],



    [[R, 0, 1, 0, G],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],

    [[R, 0, 1, 0, G],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],

    [[R, 0, 1, 0, G],
     [0, 0, 1, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],

    [[R, 0, 1, 0, G],
     [0, 0, 1, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0]],
]


class GridWorldEnv(gym.Env):
    """Gridworld environment with exposed MDP matrices (P, R, γ, μ₀)."""

    reward_goal = 1.0
    reward_step = -0.001

    def __init__(self, map=None):
        super().__init__()
        self.action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        self._init_from_map(map)
        self._build_mdp_matrices()

    def _init_from_map(self, map):
        self.map = map
        self.nrows, self.ncols = len(map), len(map[0])
        self.shape = np.array([self.nrows, self.ncols])
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.nrows * self.ncols,), dtype=np.float32)

        self.walls = np.zeros(self.shape, dtype=bool)
        self.start_pos, self.goal_pos = None, None

        for i in range(self.nrows):
            for j in range(self.ncols):
                if map[i][j] == 1: self.walls[i, j] = True
                elif map[i][j] == 'r': self.start_pos = [i, j]
                elif map[i][j] == 'g': self.goal_pos = [i, j]

        self.rewards_grid = self.reward_step * np.ones(self.shape)
        if self.goal_pos: self.rewards_grid[self.goal_pos[0], self.goal_pos[1]] = self.reward_goal
        self.terminals = self.rewards_grid > 0
        self.rowcol = np.array(self.start_pos, dtype=float) if self.start_pos else np.zeros(2)

    def _build_mdp_matrices(self):
        """Build P[s,a,s'], R[s,a], and μ₀."""
        self.n_states = self.nrows * self.ncols
        self.n_actions = 4
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))
        self.mu0 = np.zeros(self.n_states)

        if self.start_pos:
            self.mu0[self.start_pos[0] * self.ncols + self.start_pos[1]] = 1.0

        for s in range(self.n_states):
            row, col = s // self.ncols, s % self.ncols
            if self.terminals[row, col]:
                self.P[s, :, s] = 1.0
                continue
            for a, (dr, dc) in enumerate(self.action_deltas):
                nr, nc = np.clip(row + dr, 0, self.nrows - 1), np.clip(col + dc, 0, self.ncols - 1)
                if self.walls[nr, nc]: nr, nc = row, col
                ns = nr * self.ncols + nc
                self.P[s, a, ns] = 1.0
                self.R[s, a] = self.rewards_grid[nr, nc]

    def step(self, a):
        pre = self.rowcol.copy()
        self.rowcol += self.action_deltas[a]
        self.rowcol = np.clip(self.rowcol, 0, self.shape - 1).astype(int)
        if self.map[self.rowcol[0]][self.rowcol[1]] == 1: self.rowcol = pre

        obs = np.zeros(self.n_states, dtype=np.float32)
        obs[int(self.rowcol[0] * self.ncols + self.rowcol[1])] = 1
        reward = self.rewards_grid[self.rowcol[0], self.rowcol[1]]
        return obs, reward, self.terminals[self.rowcol[0], self.rowcol[1]], False, {'is_success': reward == self.reward_goal}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rowcol = np.array(self.start_pos, dtype=float) if self.start_pos else np.zeros(2)
        obs = np.zeros(self.n_states, dtype=np.float32)
        obs[int(self.rowcol[0] * self.ncols + self.rowcol[1])] = 1
        return obs, {}