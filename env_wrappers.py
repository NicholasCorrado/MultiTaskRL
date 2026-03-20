import copy
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import NormalizeObservation, NormalizeReward


class OneHotTaskWrapper(gym.ObservationWrapper):
    """
    Wraps a single-task env and prepends a fixed one-hot task encoding to observations.
    This wrapper is meant to apply to EACH TASK ENV, not the VecEnv.
    """

    def __init__(self, env: gym.Env, task_id: int, num_tasks: int, dtype=np.float32):
        super().__init__(env)
        self.task_id = int(task_id)
        self.num_tasks = int(num_tasks)
        self.dtype = dtype

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError(f"OneHotTaskWrapper requires Box obs, got {type(env.observation_space)}")

        base = env.observation_space
        self.base_shape = base.shape
        self.base_dim = int(np.prod(self.base_shape))

        low = np.concatenate([np.zeros(self.num_tasks, dtype=self.dtype), base.low.reshape(-1).astype(self.dtype)])
        high = np.concatenate([np.ones(self.num_tasks, dtype=self.dtype), base.high.reshape(-1).astype(self.dtype)])

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(self.num_tasks + self.base_dim,),
            dtype=self.dtype,
        )

    def get_task_id(self) -> int:
        return self.task_id

    def observation(self, obs):
        obs_flat = np.asarray(obs, dtype=self.dtype).reshape(-1)
        if obs_flat.shape[0] != self.base_dim:
            raise ValueError(f"Base obs dim changed: expected {self.base_dim}, got {obs_flat.shape[0]}")

        onehot = np.zeros(self.num_tasks, dtype=self.dtype)
        onehot[self.task_id] = 1.0
        return np.concatenate([onehot, obs_flat], axis=0)

class MultiTaskSampler:
    def __init__(self, probs: np.ndarray):
        self.set_probs(probs)
        self.num_tasks = len(probs)

    def set_probs(self, probs: np.ndarray) -> None:
        self.probs = probs / probs.sum()

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(len(self.probs), p=self.probs))

    def get_probs(self) -> np.ndarray:
        return self.probs.copy()

    def update(self, **kwargs):
        raise NotImplementedError

class PadObsActionWrapper(gym.Wrapper):
    """Pads observations to target_obs_dim; slices actions down to the env's original action dim."""

    def __init__(self, env: gym.Env, target_obs_dim: int, target_act_dim: int):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        self.orig_obs_dim = int(np.prod(env.observation_space.shape))
        self.orig_act_dim = int(np.prod(env.action_space.shape))

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(target_obs_dim,), dtype=np.float32
        )
        act_low  = np.concatenate([env.action_space.low.flatten(),  np.full(target_act_dim - self.orig_act_dim, -1.0, dtype=np.float32)])
        act_high = np.concatenate([env.action_space.high.flatten(), np.full(target_act_dim - self.orig_act_dim,  1.0, dtype=np.float32)])
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)

    def _pad_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return np.pad(obs, (0, self.observation_space.shape[0] - self.orig_obs_dim))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pad_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action.flat[:self.orig_act_dim])
        return self._pad_obs(obs), reward, terminated, truncated, info


class MultiTaskEnvWrapper(gym.Env):
    """
    Pools one env per task (already one-hot wrapped, each has fixed task_id).
    On reset(): sample task_id ~ p, select that env, reset it.
    step(): delegates to current env.
    """

    def __init__(
        self,
        envs: List[gym.Env],          # one env per task (already wrapped with OneHotTaskWrapper)
        sampler: MultiTaskSampler,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if len(envs) == 0:
            raise ValueError("envs must be non-empty")

        self._envs = envs
        self.sampler = sampler
        self._rng = np.random.default_rng(seed)

        self._task_id: int = 0
        self._env: gym.Env = self._envs[self._task_id]

        # Spaces must match across tasks
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        for j, e in enumerate(self._envs):
            if e.observation_space != self.observation_space:
                raise ValueError(f"Observation space mismatch at task {j}: {e.observation_space} vs {self.observation_space}")
            if e.action_space != self.action_space:
                raise ValueError(f"Action space mismatch at task {j}: {e.action_space} vs {self.action_space}")

    def set_task_probs(self, probs: np.ndarray) -> None:
        self.sampler.set_probs(probs)

    def get_task_probs(self) -> np.ndarray:
        return self.sampler.get_probs()

    def update_task_sampler(self, **kwargs):
        return self.sampler.update(**kwargs)

    def get_task_id(self) -> int:
        return int(self._task_id)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._task_id = self.sampler.sample(self._rng)
        self._env = self._envs[self._task_id]

        obs, info = self._env.reset(seed=seed, options=options)
        info = dict(info)
        # Important: propagate task_id through infos; RecordEpisodeStatistics will carry it to final_info
        info["task_id"] = self._task_id
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        info = dict(info)
        info["task_id"] = self._task_id
        return obs, reward, terminated, truncated, info

    def get_norm_stats(self):
        return [_extract_rms(e) for e in self._envs]

    def close(self):
        for e in self._envs:
            try:
                e.close()
            except Exception:
                pass

def _find_wrapper(env, cls):
    while env is not None:
        if isinstance(env, cls):
            return env
        env = getattr(env, 'env', None)
    return None

def _extract_rms(env):
    obs_w = _find_wrapper(env, NormalizeObservation)
    rew_w = _find_wrapper(env, NormalizeReward)
    return {
        'obs_rms':    copy.deepcopy(obs_w.obs_rms)    if obs_w else None,
        'return_rms': copy.deepcopy(rew_w.return_rms) if rew_w else None,
    }

def _apply_rms(env, stats):
    obs_w = _find_wrapper(env, NormalizeObservation)
    rew_w = _find_wrapper(env, NormalizeReward)
    if obs_w and stats.get('obs_rms') is not None:
        obs_w.obs_rms = copy.deepcopy(stats['obs_rms'])
    if rew_w and stats.get('return_rms') is not None:
        rew_w.return_rms = copy.deepcopy(stats['return_rms'])

def sync_task_norm_stats(train_envs, eval_envs):
    stats = train_envs.call("get_norm_stats")[0]
    for task_id, task_stats in enumerate(stats):
        obs_w = _find_wrapper(eval_envs.envs[task_id], NormalizeObservation)
        if obs_w is None:
            raise RuntimeError(
                f"NormalizeObservation not found in eval env {task_id}. "
                f"Top-level type: {type(eval_envs.envs[task_id])}"
            )
        if task_stats['obs_rms'] is None:
            raise RuntimeError(
                f"obs_rms is None for task {task_id} — "
                f"NormalizeObservation missing from training env chain?"
            )
        obs_w.obs_rms = copy.deepcopy(task_stats['obs_rms'])