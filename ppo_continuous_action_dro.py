# multitask_ppo_dro_onefile.py
# CleanRL-style PPO + DRO task reweighting, refactored to:
# - OneHotTaskWrapper applies to each per-task env (assigns fixed task_id)
# - MultiTaskEnvWrapper pools these task-envs and samples a task on reset with probs p
# - Training env is a (Sync|Async) VectorEnv over multiple copies of MultiTaskEnvWrapper
# - Eval env is a VectorEnv with one env per task (no stochastic task sampling), collecting equal episodes per task
#
# RL logic (PPO update, buffers, losses) is unchanged; only env construction + eval loop are refactored.

from __future__ import annotations

import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import custom_envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import tyro
import yaml
from stable_baselines3.common.utils import get_latest_run_id

from env_wrappers import OneHotTaskWrapper, MultiTaskEnvWrapper, MultiTaskSampler, _find_wrapper, sync_task_norm_stats, \
    PadObsActionWrapper
from task_distribution_updates import easy_first_curriculum_update, mirror_ascent_kl_update, learning_progress_update, \
    hard_first_curriculum_update

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

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    run_id: int = None
    asynchronous: bool = False  # if True, uses AsyncVectorEnv for training multitask wrappers

    # Output
    output_rootdir: str = "results"
    output_subdir: str = ""

    # Evaluation
    eval_freq: int = 10
    num_evals: int = None
    eval_episodes: int = 10

    # Multitask + DRO
    task_sampling_algo: str = 'dro'

    dro_success_ref: bool = False
    task_probs_init: List[float] = None
    dro_num_steps: int = 2048*6
    dro_eps: float = 1/6/4 # minimum task probability
    dro_eta: float = 8.0 # controls sharpness of task distribution. Larger = sharper
    dro_step_size: float = 0.1 # don't change this

    # Algorithm specific arguments
    # env_ids: List[str] = field(default_factory=lambda: [f"HardGridWorldEnv{i}-v0" for i in range(1, 4 + 1)])
    # env_ids: List[str] = field(default_factory=lambda: [f"PointMaze_UMaze-v3", "PointMaze_Medium-v3", "PointMaze_Large-v3"])
    env_ids: List[str] = field(default_factory=lambda: [f"Swimmer-v4", "HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4", "Humanoid-v4"])
    total_timesteps: int =  100_000_000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048*6
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 16
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.05

    linear: bool = False
    normalize_obs: bool = True
    normalize_reward: bool = True


    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, num_tasks, linear=True):
        super().__init__()
        self.num_tasks = num_tasks
        # Observation dim without the one-hot task encoding
        obs_dim = int(np.array(envs.single_observation_space.shape).prod()) - num_tasks
        out_dim = int(np.prod(envs.single_action_space.shape))

        if linear:
            self.critic_trunk = None
            self.actor_trunk = None
            trunk_dim = obs_dim
        else:
            self.critic_trunk = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
            )
            self.actor_trunk = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
            )
            trunk_dim = 256

        # Multi-headed outputs: (num_tasks, trunk_dim, out_dim)
        self.critic_heads = nn.Parameter(torch.zeros(num_tasks, trunk_dim, 1))
        self.actor_heads = nn.Parameter(torch.zeros(num_tasks, trunk_dim, out_dim))

        # Initialize heads
        for i in range(num_tasks):
            nn.init.orthogonal_(self.critic_heads[i], gain=1.0)
            nn.init.orthogonal_(self.actor_heads[i], gain=0.01)

        # Per-task log std
        self.actor_logstd = nn.Parameter(torch.zeros(num_tasks, out_dim))

    def _split_obs(self, x):
        """Split observation into task one-hot and actual observation."""
        task_onehot = x[..., :self.num_tasks]
        obs = x[..., self.num_tasks:]
        return task_onehot, obs

    def _apply_heads(self, features, heads, task_onehot):
        """Efficiently apply task-specific heads using einsum.

        features: (batch, trunk_dim)
        heads: (num_tasks, trunk_dim, out_dim)
        task_onehot: (batch, num_tasks)

        Returns: (batch, out_dim)
        """
        # Compute all head outputs: (batch, num_tasks, out_dim)
        all_outputs = torch.einsum('bf,tfo->bto', features, heads)
        # Select using one-hot: (batch, out_dim)
        # task_onehot is (batch, num_tasks), need to expand for broadcasting
        return torch.einsum('bt,bto->bo', task_onehot, all_outputs)

    def get_value(self, x):
        task_onehot, obs = self._split_obs(x)
        features = self.critic_trunk(obs) if self.critic_trunk else obs
        return self._apply_heads(features, self.critic_heads, task_onehot)

    def get_action_and_value(self, x, action=None):
        task_onehot, obs = self._split_obs(x)

        actor_features = self.actor_trunk(obs) if self.actor_trunk else obs
        critic_features = self.critic_trunk(obs) if self.critic_trunk else obs

        action_mean = self._apply_heads(actor_features, self.actor_heads, task_onehot)
        value = self._apply_heads(critic_features, self.critic_heads, task_onehot)

        # Select task-specific logstd: (batch, out_dim)
        action_logstd = torch.einsum('bt,to->bo', task_onehot, self.actor_logstd)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), value

    def get_action(self, x, sample=True):
        task_onehot, obs = self._split_obs(x)
        actor_features = self.actor_trunk(obs) if self.actor_trunk else obs
        action_mean = self._apply_heads(actor_features, self.actor_heads, task_onehot)

        if sample:
            action_logstd = torch.einsum('bt,to->bo', task_onehot, self.actor_logstd)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            return probs.sample()
        return action_mean

class Agent(nn.Module):
    def __init__(self, envs, linear=True):
        super().__init__()
        in_dim = int(np.array(envs.single_observation_space.shape).prod())
        out_dim = int(np.prod(envs.single_action_space.shape))

        if linear:
            self.critic = nn.Sequential(layer_init(nn.Linear(in_dim, 1), std=1.0))
            self.actor_mean = nn.Sequential(layer_init(nn.Linear(in_dim, out_dim), std=0.01))
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(in_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(in_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, out_dim), std=0.01),
            )

        self.actor_logstd = nn.Parameter(torch.zeros(1, out_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)

    def get_action(self, x, sample=True):
        action_mean = self.actor_mean(x)
        if sample:
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            return probs.sample()
        return action_mean


def exponentiated_gradient_ascent_step(
    args: Args,
    w: np.ndarray,
    returns: np.ndarray,
    returns_ref: np.ndarray,
    previous_return_avg: np.ndarray,
    learning_rate: float = 1.0,
    eps: float = 0.1,
) -> np.ndarray:

    diff = np.clip(returns_ref - returns, 0, np.inf)
    w_new = w * np.exp(learning_rate * diff)
    w_new = w_new / w_new.sum()

    w_uniform = np.ones_like(w_new) / len(w_new)
    w_new = (1 - eps) * w_new + eps * w_uniform
    return w_new


# ----------------------------
# Env construction utilities
# ----------------------------

def make_base_env(env_id, capture_video, video_path, render,
                  normalize_obs=False, normalize_reward=False, gamma=0.99):
    if capture_video and render:
        env = gym.make(env_id, render_mode="rgb_array", continuing_task=False, reset_target=False)
        env = gym.wrappers.RecordVideo(env, video_path)
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)  # since humaniod has different action bounds than other tasks
    env = gym.wrappers.ClipAction(env)
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

def make_multitask_train_vec_env(
    env_ids: List[str],
    num_envs: int,
    capture_video: bool,
    run_name: str,
    task_probs_init: Optional[List[float]] = None,
    seed: Optional[int] = None,
    asynchronous: bool = False,
    normalize_obs=False,
    normalize_reward=False,
    gamma=0.99
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
                e = make_base_env(env_id, capture_video=capture_video, video_path=video_path, render=render,
                                  normalize_obs=normalize_obs, normalize_reward=normalize_reward, gamma=gamma)

                target_obs_dim = 0
                target_act_dim = 0
                for env_id in env_ids:
                    e_tmp = make_base_env(env_id, capture_video=False, video_path="", render=False)
                    target_obs_dim = max(target_obs_dim, int(np.prod(e_tmp.observation_space.shape)))
                    target_act_dim = max(target_act_dim, int(np.prod(e_tmp.action_space.shape)))
                    e_tmp.close()

                e = PadObsActionWrapper(e, target_obs_dim, target_act_dim)
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

    def get_task_id() -> np.ndarray:
        ps = envs.call("get_task_id")
        return ps[0]

    def set_task_probs(p: np.ndarray) -> None:
        p = np.asarray(p, dtype=np.float64)
        p = p / p.sum()
        envs.call("set_task_probs", p)

    envs.get_task_id = get_task_id
    envs.get_task_probs = get_task_probs  # type: ignore[attr-defined]
    envs.set_task_probs = set_task_probs  # type: ignore[attr-defined]
    return envs


def make_eval_vec_env(
    env_ids: List[str],
    capture_video: bool,
    run_name: str,
    asynchronous: bool = True,
    normalize_obs=False,
    normalize_reward=False,
    gamma=0.99
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
            e = make_base_env(env_ids[task_id], capture_video=capture_video, video_path=video_path, render=render,
                              normalize_obs=normalize_obs, normalize_reward=False, gamma=gamma)

            target_obs_dim = 0
            target_act_dim = 0
            for env_id in env_ids:
                e_tmp = make_base_env(env_id, capture_video=False, video_path="", render=False)
                target_obs_dim = max(target_obs_dim, int(np.prod(e_tmp.observation_space.shape)))
                target_act_dim = max(target_act_dim, int(np.prod(e_tmp.action_space.shape)))
                e_tmp.close()

            e = PadObsActionWrapper(e, target_obs_dim, target_act_dim)
            e = OneHotTaskWrapper(e, task_id=task_id, num_tasks=num_tasks)

            return e

        return _thunk

    thunks = [thunk(t) for t in range(num_tasks)]
    return gym.vector.AsyncVectorEnv(thunks) if asynchronous else gym.vector.SyncVectorEnv(thunks)


def simulate_equal_episodes_per_task(envs_eval, actor: Agent, eval_episodes_per_task: int):
    """
    envs_eval has num_envs == num_tasks, one env per task, so this ensures equal episode counts per task.
    Returns: (return_avg, return_std, success_avg, success_std) per task.
    """
    num_tasks = envs_eval.num_envs
    returns_by_task = [[] for _ in range(num_tasks)]
    success_by_task = [[] for _ in range(num_tasks)]
    counts = np.zeros(num_tasks, dtype=np.int32)

    obs, _ = envs_eval.reset()
    while np.any(counts < eval_episodes_per_task):
        with torch.no_grad():
            actions = actor.get_action(torch.as_tensor(obs).to("cpu"), sample=False).cpu().numpy()

        obs, _, _, _, infos = envs_eval.step(actions)

        if "final_info" in infos:
            for i, finfo in enumerate(infos["final_info"]):
                if finfo and "episode" in finfo:
                    if counts[i] < eval_episodes_per_task:
                        returns_by_task[i].append(float(finfo["episode"]["r"]))
                        success_by_task[i].append(float(finfo.get("is_success", finfo.get("success", 0.0))))
                        counts[i] += 1

    per_task = []
    for i in range(num_tasks):
        r = np.asarray(returns_by_task[i], dtype=np.float64)
        s = np.asarray(success_by_task[i], dtype=np.float64)
        per_task.append((
            float(r.mean()), float(r.std()),
            float(s.mean()), float(s.std()),
        ))
    return per_task


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    env_name = "_".join(args.env_ids)
    env_name = "GridWorld4"
    run_name = f"{env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.num_evals is not None:
        args.eval_freq = args.num_iterations // args.num_evals

    # Seeding (kept as-is; note your original only seeds if args.seed is None)
    if args.seed is None:
        if args.run_id:
            args.seed = args.run_id
        else:
            args.seed = np.random.randint(2**32 - 1)

            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = args.torch_deterministic

    # Output path
    args.output_dir = f"{args.output_rootdir}/{env_name}/ppo/{args.output_subdir}"
    if args.run_id is not None:
        args.output_dir += f"/run_{args.run_id}"
    else:
        run_id = get_latest_run_id(log_path=args.output_dir, log_name="run") + 1
        args.output_dir += f"/run_{run_id}"
    print(f"output_dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ----------------------------
    # Env setup (refactored only)
    # ----------------------------
    num_tasks = len(args.env_ids)

    envs = make_multitask_train_vec_env(
        env_ids=args.env_ids,
        num_envs=args.num_envs,
        capture_video=args.capture_video,
        run_name=run_name,
        task_probs_init=args.task_probs_init,
        seed=args.seed,
        asynchronous=args.asynchronous,
        normalize_obs=args.normalize_obs,
        normalize_reward=args.normalize_reward,
        gamma=args.gamma,
    )

    # Deterministic eval vec env: one env per task
    envs_eval = make_eval_vec_env(
        env_ids=args.env_ids,
        capture_video=args.capture_video,
        run_name=run_name,
        asynchronous=False,
        normalize_obs=args.normalize_obs,
        normalize_reward=args.normalize_reward,
        gamma=args.gamma,
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, linear=args.linear).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup (unchanged)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    update_count = 0
    logs = defaultdict(lambda: [])

    training_returns = [[] for _ in range(num_tasks)]
    training_returns_avg_prev = None

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    current_task_distribution = envs.get_task_probs()
    current_task_objective_weights = np.ones(num_tasks) / num_tasks
    current_task_id = envs.get_task_id()

    is_task_solved = np.zeros(num_tasks)

    task_buffer_length = 30
    task_success_buffer = [deque(maxlen=task_buffer_length) for i in range(num_tasks)]

    max_returns = np.zeros(num_tasks)
    max_returns[:] = -np.inf
    max_velocity = np.zeros(num_tasks)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)

            if "final_info" in infos:
                for finfo in infos["final_info"]:
                    if finfo and "episode" in finfo:
                        tid = int(finfo["task_id"])
                        if args.dro_success_ref:
                            training_returns[tid].append(float(finfo.get("is_success", 0.0)))
                        else:
                            training_returns[tid].append(float(finfo["episode"]["r"]))
                            if max_returns[tid] < float(finfo["episode"]["r"]):
                                max_returns[tid] = float(finfo["episode"]["r"])
                        task_success_buffer[tid].append(float(finfo.get("is_success", 0.0)))
                        current_task_id = envs.get_task_id()

                        max_velocity[current_task_id] = max(max_velocity[current_task_id], finfo['x_velocity'])
            else:
                max_velocity[current_task_id] = max(max_velocity[current_task_id], infos['x_velocity'])

            if global_step % args.dro_num_steps == 0:
                training_returns_avg = np.array([np.mean(training_returns[i]) if len(training_returns[i]) > 0 else 0 for i in range(num_tasks)])
                returns_ref = max_returns
                # print(max_velocity)
                # w_forward = np.array([1, 1, 1, 1, 1, 1.25])
                # returns_ref = max_velocity * w_forward * 1000

                return_gap = np.clip(returns_ref - training_returns_avg, -np.inf, returns_ref) / returns_ref
                return_slope = np.abs(training_returns_avg - training_returns_avg_prev) if training_returns_avg_prev is not None else np.zeros(num_tasks)
                # print(returns_ref)
                # print(return_gap)

                success_rates = np.array([
                    np.mean(task_success_buffer[i]) if len(task_success_buffer[i]) >= task_buffer_length else 0.0
                    for i in range(num_tasks)
                ])

                if args.task_sampling_algo == 'uniform':
                    pass
                elif args.task_sampling_algo == 'dro':
                    current_task_distribution = mirror_ascent_kl_update(
                        q=current_task_distribution,
                        gap=return_gap,
                        eta=args.dro_eta,
                        step_size=args.dro_step_size,
                        eps=args.dro_eps,
                        p0=None,  # uniform by default
                    )
                elif args.task_sampling_algo == 'dro_reweight':
                    current_task_objective_weights = mirror_ascent_kl_update(
                        q=current_task_objective_weights,
                        gap=return_gap,
                        eta=args.dro_eta,
                        step_size=args.dro_step_size,
                        eps=args.dro_eps,
                        p0=None,  # uniform by default
                    )
                elif args.task_sampling_algo == 'learning_progress':
                    current_task_distribution = learning_progress_update(
                        q=current_task_distribution,
                        learning_progress=return_slope,
                        eta=args.dro_eta,
                        step_size=args.dro_step_size,
                        eps=args.dro_eps,
                        p0=None,  # uniform by default
                        success_rates=success_rates,
                        success_threshold=0.9,
                    )
                elif args.task_sampling_algo == 'easy_first':
                    current_task_distribution = easy_first_curriculum_update(
                        success_rates=success_rates,
                        success_threshold=0.9,
                        eps=args.dro_eps,  # now interpreted as floor min prob
                    )
                elif args.task_sampling_algo == 'hard_first':
                    current_task_distribution = hard_first_curriculum_update(
                        success_rates=success_rates,
                        success_threshold=0.9,
                        eps=args.dro_eps,  # now interpreted as floor min prob
                    )

                envs.set_task_probs(current_task_distribution)
                # print(current_task_distribution)

                training_returns_avg_prev = training_returns_avg
                training_returns = [[] for _ in range(num_tasks)]

        # bootstrap value if not done (unchanged)
        with torch.no_grad():
            next_value = values[-1].reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch (unchanged)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_task_ids = b_obs[:, :num_tasks].argmax(dim=-1)

        # PPO update (unchanged)
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                # if args.norm_adv:
                #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                if args.norm_adv:
                    # Compute per-task means and counts
                    task_ids = b_task_ids[mb_inds]
                    counts = torch.bincount(task_ids, minlength=num_tasks).float()  # (num_tasks,)
                    means = torch.zeros(num_tasks, device=mb_advantages.device)
                    means.scatter_add_(0, task_ids, mb_advantages)
                    means /= counts.clamp(min=1)

                    # Compute per-task stds
                    sq_diff = (mb_advantages - means[task_ids]) ** 2
                    vars_ = torch.zeros(num_tasks, device=mb_advantages.device)
                    vars_.scatter_add_(0, task_ids, sq_diff)
                    vars_ /= (counts - 1).clamp(min=1)
                    stds = vars_.sqrt()

                    # Normalize
                    valid = counts[task_ids] > 1
                    mb_advantages = torch.where(
                        valid,
                        (mb_advantages - means[task_ids]) / (stds[task_ids] + 1e-8),
                        mb_advantages
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_count += 1

        # Eval (refactored only: deterministic, equal episodes per task)
        if iteration % args.eval_freq == 0:
            print(f"Eval num_timesteps={global_step}")
            p_now = envs.get_task_probs()
            print(f"Task probs: {p_now}")

            logs["timestep"].append(global_step)
            logs["update"].append(update_count)

            sync_task_norm_stats(envs, envs_eval)

            per_task_stats = simulate_equal_episodes_per_task(envs_eval, agent, eval_episodes_per_task=args.eval_episodes)

            return_all_tasks = []
            success_all_tasks = []

            for j, (return_avg, return_std, success_avg, success_std) in enumerate(per_task_stats):
                return_all_tasks.append(return_avg)
                success_all_tasks.append(success_avg)

                print(f"Task {j}: {args.env_ids[j]}")
                print(f"episode_return={return_avg:.2f} +/- {return_std:.2f}")
                print(f"episode_success={success_avg:.2f} +/- {success_std:.2f}")
                print()

                logs[f"task_probs_{j}"].append(p_now[j])
                logs[f"return_{j}"].append(return_avg)
                logs[f"success_rate_{j}"].append(success_avg)

            return_all_tasks_avg = float(np.mean(return_all_tasks))
            success_all_tasks_avg = float(np.mean(success_all_tasks))

            print("Average over all tasks:")
            print(f"episode_return={return_all_tasks_avg:.2f} +/- (per-task stds shown above)")
            print(f"episode_success={success_all_tasks_avg:.2f}")
            print()

            logs["return"].append(return_all_tasks_avg)
            logs["success_rate"].append(success_all_tasks_avg)

            np.savez(f"{args.output_dir}/evaluations.npz", **logs)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    envs_eval.close()