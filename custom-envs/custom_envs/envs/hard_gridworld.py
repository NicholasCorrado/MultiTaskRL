from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

gridworld_maps = [
    # Map 1
    # Expected Steps Num from r to g: 7
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 'r', 0, 0, 0, 0, 0, 0, 'g', 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

    # Map 2
    # Expected Steps Num from r to g: 14
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 'r', 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 'g', 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

    # Map 3
    # Expected Steps Num from r to g: 21
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 'r', 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 'g', 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

    # Map 4
    # Expected Steps Num from r to g: 21
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 'r', 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 'g', 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
]

n = 3

class HardGridWorldEnv(gym.Env):
    def __init__(self, shape=(5,5), rewards=(-0.01, 0.5, 1), map=None):
        super().__init__()

        self.rewards = rewards

        # Default GridWorld map, reborn in the middle, most optimal goal on the right-bottom corner, sub optimal goal on the left-top corner.
        if map is None:
            self.map_on = False

            self.shape = np.array(shape)
            self.action_space = gym.spaces.Discrete(4)

            obs_dim = n + shape[0] * shape[1]
            self.observation_space = gym.spaces.Box(low=np.zeros(obs_dim), high=np.ones(obs_dim), shape=(obs_dim,))

            self.nrows, self.ncols = self.shape
            self.rowcol = (self.shape-1)/2
            self.init_rowcol = (self.shape-1)/2 # agent starts in middle of the grid.

            self.rewards = rewards[0] * np.ones(shape=self.shape)
            self.rewards[0, 0] = rewards[1] # subopt
            self.rewards[self.nrows-1, self.ncols-1] = rewards[2]
            self.opt_reward = rewards[2]

            self.terminals = np.zeros(shape=self.shape, dtype=bool)
            self.terminals[self.rewards > 0] = True

            self.task_id = np.zeros(4)

            self.map = np.zeros(self.shape)

        else:
            # With a given map: 'r' where the agent is reborn, 'g' is the goal, 0 is empty position, 1 is wall.
            self.map_on = True
            self.map = map

            # Set shape to map's shape
            shape = [len(map), len(map[0])]
            shape = np.array(shape)
            self.shape = shape
            self.action_space = gym.spaces.Discrete(4)

            obs_dim = n + shape[0] * shape[1]
            self.observation_space = gym.spaces.Box(low=np.zeros(obs_dim), high=np.ones(obs_dim), shape=(obs_dim,))

            # The reborn/goal position of this map
            self.start_pos, self.end_pos = None, None

            # Positions with blank cells
            self.free_pos = []
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if map[i][j] == 'r':
                        self.start_pos = [i, j]
                    if map[i][j] == 'g':
                        self.end_pos = [i, j]
                    if map[i][j] == 0:
                        self.free_pos.append([i, j])

            if self.end_pos is None:
                self.end_pos = np.random.choice(self.free_pos)

            self.rewards = rewards[0] * np.ones(shape=self.shape)
            self.rewards[self.end_pos[0], self.end_pos[1]] = rewards[2]
            self.opt_reward = rewards[2]

            self.terminals = np.zeros(shape=self.shape, dtype=bool)
            self.terminals[self.rewards > 0] = True

            self.task_id = np.zeros(8)

        print(self.rewards)
        print(self.terminals)


    def _rowcol_to_obs(self, rowcol):
        idx = int(rowcol[0] * self.shape[0] + rowcol[1])
        state = np.zeros(self.shape[0] * self.shape[1])
        state[idx] = 1
        return state

    def step(self, a):
        pre_pos = self.rowcol.copy()
        # up
        if a == 0:
            self.rowcol[0] -= 1
        # down
        elif a == 1:
            self.rowcol[0] += 1
        # left
        elif a == 2:
            self.rowcol[1] -= 1
        # down
        elif a == 3:
            self.rowcol[1] += 1

        # print ("Shape:", self.shape)
        # print ("Start Position: ", self.start_pos)
        # print ("A:", self.rowcol)
        self.rowcol = np.clip(self.rowcol, a_min=np.zeros(2), a_max=self.shape-1).astype(int)
        # print ("B:", self.rowcol)

        if self.map[self.rowcol[0]][self.rowcol[1]] == 1:
            self.rowcol = pre_pos

        state = self._rowcol_to_obs(self.rowcol)
        # print ("C:", self.rowcol)
        # print ("D:", self.rewards)
        reward = self.rewards[self.rowcol[0], self.rowcol[1]]
        terminated = self.terminals[self.rowcol[0], self.rowcol[1]]
        truncated = False
        info = {'is_success': reward == self.opt_reward}

        return np.concatenate([self.task_id, state]), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if self.map_on:
            if self.start_pos != -1:
                self.rowcol = self.start_pos.copy()
            else:
                self.rowcol = np.random.choice(self.free_pos)
        else:
            self.rowcol = self.init_rowcol.copy()
        state = self._rowcol_to_obs(self.rowcol)

        return np.concatenate([self.task_id, state]), {}


class HardGridWorldEnv1(HardGridWorldEnv):
    def __init__(self, shape=(10, 10), rewards=(-0.01, 0, 1)):
        map = gridworld_maps[0]
        super().__init__(shape=(len(map), len(map[0])), rewards=(-0.01, 0, 1), map=map)
        self.task_id = np.zeros(n)
        self.task_id[0] = 1

class HardGridWorldEnv2(HardGridWorldEnv):
    def __init__(self, shape=(10, 10), rewards=(-0.01, 0, 1)):
        map = gridworld_maps[1]
        super().__init__(shape=(len(map), len(map[0])), rewards=(-0.01, 0, 1), map=map)
        self.task_id = np.zeros(n)
        self.task_id[1] = 1

class HardGridWorldEnv3(HardGridWorldEnv):
    def __init__(self, shape=(10, 10), rewards=(-0.01, 0, 1)):
        map = gridworld_maps[2]
        super().__init__(shape=(len(map), len(map[0])), rewards=(-0.01, 0, 1), map=map)
        self.task_id = np.zeros(n)
        self.task_id[2] = 1

# class HardGridWorldEnv4(HardGridWorldEnv):
#     def __init__(self, shape=(10, 10), rewards=(-0.01, 0, 1)):
#         super().__init__(shape=(10, 10), rewards=(-0.01, 0, 1), map=gridworld_maps[3])
#         self.task_id = np.zeros(n)
#         self.task_id[3] = 1