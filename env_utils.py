from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

@dataclass
class MultiTaskSampler:
    """Holds p and samples task indices."""
    probs: np.ndarray

    def __post_init__(self):
        self.set_probs(self.probs)

    def set_probs(self, probs: np.ndarray) -> None:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 1:
            raise ValueError(f"probs must be 1D, got shape={probs.shape}")
        if np.any(probs < 0):
            raise ValueError("probs must be non-negative")
        s = probs.sum()
        if s <= 0:
            raise ValueError("probs must sum to a positive value")
        self.probs = probs / s

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(len(self.probs), p=self.probs))

    def get_probs(self) -> np.ndarray:
        return self.probs.copy()


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

    def close(self):
        for e in self._envs:
            try:
                e.close()
            except Exception:
                pass

# ----------------------------
# Env construction utilities
# ----------------------------

def make_base_env(env_id: str, capture_video: bool, video_path: str, render: bool) -> gym.Env:
    if capture_video and render:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_path)
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def make_multitask_train_vec_env(
    env_ids: List[str],
    num_envs: int,
    capture_video: bool,
    run_name: str,
    task_probs_init: Optional[List[float]] = None,
    seed: Optional[int] = None,
    asynchronous: bool = False,
):
    """
    Returns a VecEnv whose elements are MultiTaskEnvWrapper instances.
    Each MultiTaskEnvWrapper pools one env per task; each per-task env is OneHotTaskWrapper(task_id=j).
    """
    num_tasks = len(env_ids)
    if task_probs_init is None:
        probs0 = np.ones(num_tasks, dtype=np.float64) / num_tasks
    else:
        probs0 = np.asarray(task_probs_init, dtype=np.float64)
        probs0 = probs0 / probs0.sum()

    def slot_thunk(slot_idx: int):
        def _thunk():
            # Each slot has its own sampler (works for Sync and Async).
            sampler = MultiTaskSampler(probs=probs0.copy())

            task_envs: List[gym.Env] = []
            for task_id, env_id in enumerate(env_ids):
                # Apply onehot at the task/env level (fixed task_id)
                render = (slot_idx == 0 and task_id == 0)  # only record one video stream
                video_path = f"videos/{run_name}"
                e = make_base_env(env_id, capture_video=capture_video, video_path=video_path, render=render)
                e = OneHotTaskWrapper(e, task_id=task_id, num_tasks=num_tasks)
                task_envs.append(e)

            env = MultiTaskEnvWrapper(task_envs, sampler=sampler, seed=None if seed is None else seed + slot_idx)
            return env

        return _thunk

    thunks = [slot_thunk(i) for i in range(num_envs)]
    envs = gym.vector.AsyncVectorEnv(thunks) if asynchronous else gym.vector.SyncVectorEnv(thunks)

    # Provide vector-level get/set that works for both Sync/Async.
    # We read/write through envs.call so it works in Async too.
    def get_task_probs() -> np.ndarray:
        ps = envs.call("get_task_probs")
        return np.asarray(ps[0], dtype=np.float64)

    def set_task_probs(p: np.ndarray) -> None:
        p = np.asarray(p, dtype=np.float64)
        p = p / p.sum()
        envs.call("set_task_probs", p)

    envs.get_task_probs = get_task_probs  # type: ignore[attr-defined]
    envs.set_task_probs = set_task_probs  # type: ignore[attr-defined]
    return envs


def make_eval_vec_env(
    env_ids: List[str],
    capture_video: bool,
    run_name: str,
    asynchronous: bool = False,
):
    """
    Deterministic eval: VecEnv with one env per task.
    No random task sampling. Each env is a single task with a fixed one-hot id.
    """
    num_tasks = len(env_ids)

    def thunk(task_id: int):
        def _thunk():
            render = (task_id == 0)
            video_path = f"videos/{run_name}_eval"
            e = make_base_env(env_ids[task_id], capture_video=capture_video, video_path=video_path, render=render)
            e = OneHotTaskWrapper(e, task_id=task_id, num_tasks=num_tasks)
            return e

        return _thunk

    thunks = [thunk(t) for t in range(num_tasks)]
    return gym.vector.AsyncVectorEnv(thunks) if asynchronous else gym.vector.SyncVectorEnv(thunks)
