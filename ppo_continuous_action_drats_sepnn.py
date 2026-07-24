"""Continuous-control multi-task PPO with a separate network per task.

This is the "no parameter sharing" end of the multi-task architecture spectrum:
each task gets its own actor MLP, its own critic MLP, and its own log-std, and
no weights are shared between tasks. The task one-hot is therefore *stripped*
from the observation before it reaches a network -- it is only used to route
each sample to its task's network.

Why run it: with separate networks there is no positive transfer and no
gradient conflict, so any remaining difference between task-sampling methods is
attributable to data allocation alone. It is also the natural control for the
capacity confound, since k separate networks have k times the parameters of the
shared network used in `ppo_continuous_action_dro.py`.

Everything else (task sampling, the return-gap estimator, evaluation, logging)
is shared with the DDPG/TD3 DRATS scripts, so architecture is the only
difference from the shared-network runs.

Baselines available through --task_sampling_algo:
    uniform             fixed uniform task distribution
    drats               DRATS (alias: dro)
    drats_reweight      DRO on the *objective* instead of the sampler: tasks are
                        sampled uniformly and the PPO losses are reweighted by q
                        (alias: dro_reweight)
    learning_progress   ALP-GMM / RIAC: |J_hat_t - J_hat_{t-1}|
    learning_potential  PLR: mean |GAE advantage| per task
    hard_first          SMT: uniform over the K hardest unsolved tasks
    easy_first          easiest unsolved task first
"""

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import custom_envs  # noqa: F401
import gymnasium_robotics  # noqa: F401  # registers PointMaze_*-v3 on import (needed on CHTC)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from stable_baselines3.common.utils import get_latest_run_id
from torch.distributions.normal import Normal

from env_wrappers import MultiTaskEnvWrapper, MultiTaskSampler, OneHotTaskWrapper
from return_gap import ReturnGapEstimator
from task_distribution_updates import (
    easy_first_curriculum_update,
    hard_first_curriculum_update,
    learning_progress_update,
    mirror_ascent_kl_update,
)

DRATS_ALIASES = {"dro", "drats"}
REWEIGHT_ALIASES = {"dro_reweight", "drats_reweight"}


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = None
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False

    run_id: int = None
    asynchronous: bool = True
    """if True, uses AsyncVectorEnv for the training envs"""

    # Output
    output_rootdir: str = "results"
    output_subdir: str = ""
    env_name: str = None
    """name of the benchmark used in the output path; defaults to the joined env ids"""

    # Evaluation
    eval_freq: int = 50
    """evaluate every eval_freq PPO iterations; overridden by num_evals if set"""
    num_evals: int = None
    eval_episodes: int = 100
    """number of evaluation episodes per task"""

    # Multitask task sampling
    task_sampling_algo: str = "drats"
    """uniform | drats | drats_reweight | learning_progress | learning_potential | hard_first | easy_first"""
    task_probs_init: List[float] = None
    dro_num_steps: int = 4096
    """environment steps between task-distribution updates"""
    dro_eps: float = 0.01
    """minimum sampling probability (KL projection floor)"""
    dro_eta: float = 16.0
    """inverse temperature; larger = sharper task distribution"""
    dro_step_size: float = 0.2
    """mirror ascent step size, in [0, 1]"""
    dro_return_window: int = 20
    """number of recent episodes per task averaged into the MC return estimate"""
    dro_reset_returns: bool = False
    """if True, discard buffered returns after each update (paper's per-iteration MC estimate)"""
    dro_success_ref: bool = False
    """if True, compute gaps on success rates (J_ref = 1) instead of returns"""
    success_window: int = 20
    """number of recent episodes per task averaged into the success rate"""
    success_threshold: float = 0.9
    """success rate above which the easy/hard-first curricula consider a task solved"""
    return_ref_init: List[float] = None
    """initial reference return, e.g. r_max * H_max; None = use the max observed return"""
    return_rand: List[float] = None
    """random-policy return used to normalize gaps; None = estimate it online"""
    dro_rand_window: int = 10
    """episodes per task used to estimate J_rand from the untrained initial policy"""
    ref_success_threshold: float = 0.5
    """success rate above which J_ref switches to the max observed return"""

    # Architecture
    hidden_dim: int = 256
    """width of each task's actor/critic MLP"""
    linear: bool = False
    """use a linear actor/critic per task instead of an MLP"""
    clip_grad_per_task: bool = True
    """clip each task's subnetwork separately, so one task's large gradient cannot
    shrink another's update. Set False to clip the global norm like the shared-network runs."""

    # Algorithm specific arguments
    env_ids: List[str] = field(default_factory=lambda: ["PointMaze_UMaze-v3", "PointMaze_Medium-v3", "PointMaze_Large-v3"])
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 4096
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 16
    norm_adv: bool = False
    """normalize advantages over the whole batch (ignored when norm_adv_per_task is set)"""
    norm_adv_per_task: bool = True
    """normalize advantages within each task, as in the DRATS experiments"""
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.03
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.05

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int, out_std: float, linear: bool) -> nn.Sequential:
    if linear:
        return nn.Sequential(layer_init(nn.Linear(in_dim, out_dim), std=out_std))
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden_dim)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_dim, hidden_dim)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_dim, out_dim), std=out_std),
    )


class Agent(nn.Module):
    """One independent actor, critic, and log-std per task. No shared weights."""

    def __init__(self, envs, num_tasks: int, hidden_dim: int = 256, linear: bool = False):
        super().__init__()
        self.num_tasks = int(num_tasks)
        # The one-hot prefix is only a routing key, so each task's network sees
        # the raw observation without it.
        self.obs_dim = int(np.array(envs.single_observation_space.shape).prod()) - self.num_tasks
        self.act_dim = int(np.prod(envs.single_action_space.shape))

        self.critics = nn.ModuleList(
            [make_mlp(self.obs_dim, 1, hidden_dim, out_std=1.0, linear=linear) for _ in range(self.num_tasks)]
        )
        self.actor_means = nn.ModuleList(
            [make_mlp(self.obs_dim, self.act_dim, hidden_dim, out_std=0.01, linear=linear) for _ in range(self.num_tasks)]
        )
        # A ParameterList (not one (k, act_dim) tensor) so each task's parameters
        # form a disjoint group for per-task gradient clipping.
        self.actor_logstds = nn.ParameterList([nn.Parameter(torch.zeros(self.act_dim)) for _ in range(self.num_tasks)])

    def task_parameters(self, task_id: int):
        """All parameters belonging to one task, as a flat list."""
        return (
            list(self.actor_means[task_id].parameters())
            + list(self.critics[task_id].parameters())
            + [self.actor_logstds[task_id]]
        )

    def _split_obs(self, x):
        task_ids = x[..., : self.num_tasks].argmax(dim=-1)
        return task_ids, x[..., self.num_tasks :]

    def _dispatch(self, obs, task_ids, modules, out_dim):
        """Run each row through its own task's network and reassemble the batch."""
        outs, idxs = [], []
        for t in range(self.num_tasks):
            idx = (task_ids == t).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            idxs.append(idx)
            outs.append(modules[t](obs.index_select(0, idx)))

        out = torch.zeros(obs.shape[0], out_dim, device=obs.device, dtype=obs.dtype)
        if not idxs:
            return out
        return out.index_copy(0, torch.cat(idxs), torch.cat(outs))

    def _logstd(self, task_ids):
        return torch.stack(list(self.actor_logstds))[task_ids]

    def get_value(self, x):
        task_ids, obs = self._split_obs(x)
        return self._dispatch(obs, task_ids, self.critics, 1)

    def get_action_and_value(self, x, action=None):
        task_ids, obs = self._split_obs(x)
        action_mean = self._dispatch(obs, task_ids, self.actor_means, self.act_dim)
        action_std = torch.exp(self._logstd(task_ids))
        value = self._dispatch(obs, task_ids, self.critics, 1)

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), value

    def get_action(self, x, sample=True):
        task_ids, obs = self._split_obs(x)
        action_mean = self._dispatch(obs, task_ids, self.actor_means, self.act_dim)
        if not sample:
            return action_mean
        action_std = torch.exp(self._logstd(task_ids))
        return Normal(action_mean, action_std).sample()


# ----------------------------
# Env construction
# ----------------------------

def episode_scalar(x) -> float:
    """RecordEpisodeStatistics stores per-env arrays; reduce to a plain float."""
    return float(np.mean(x))


def make_base_env(env_id: str, capture_video: bool, video_path: str, render: bool) -> gym.Env:
    if capture_video and render:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_path)
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FlattenObservation(env)
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
    """VecEnv of MultiTaskEnvWrappers, each of which samples a task from q on reset."""
    num_tasks = len(env_ids)
    if task_probs_init is None:
        probs0 = np.ones(num_tasks, dtype=np.float64) / num_tasks
    else:
        probs0 = np.asarray(task_probs_init, dtype=np.float64)
        probs0 = probs0 / probs0.sum()

    def slot_thunk(slot_idx: int):
        def _thunk():
            # Each slot owns its sampler so this works for Sync and Async alike.
            sampler = MultiTaskSampler(probs0.copy())

            task_envs = []
            for task_id, env_id in enumerate(env_ids):
                render = slot_idx == 0 and task_id == 0
                env = make_base_env(env_id, capture_video, f"videos/{run_name}", render)
                env = OneHotTaskWrapper(env, task_id=task_id, num_tasks=num_tasks)
                task_envs.append(env)

            return MultiTaskEnvWrapper(task_envs, sampler=sampler, seed=None if seed is None else seed + slot_idx)

        return _thunk

    thunks = [slot_thunk(i) for i in range(num_envs)]
    envs = gym.vector.AsyncVectorEnv(thunks) if asynchronous else gym.vector.SyncVectorEnv(thunks)

    def get_task_probs() -> np.ndarray:
        return np.asarray(envs.call("get_task_probs")[0], dtype=np.float64)

    def set_task_probs(p: np.ndarray) -> None:
        p = np.asarray(p, dtype=np.float64)
        envs.call("set_task_probs", p / p.sum())

    envs.get_task_probs = get_task_probs  # type: ignore[attr-defined]
    envs.set_task_probs = set_task_probs  # type: ignore[attr-defined]
    return envs


def make_eval_vec_env(env_ids: List[str], asynchronous: bool = False):
    """One env per task, no task sampling, so evaluation is balanced by construction."""
    num_tasks = len(env_ids)

    def thunk(task_id: int):
        def _thunk():
            env = make_base_env(env_ids[task_id], capture_video=False, video_path="", render=False)
            return OneHotTaskWrapper(env, task_id=task_id, num_tasks=num_tasks)

        return _thunk

    thunks = [thunk(t) for t in range(num_tasks)]
    return gym.vector.AsyncVectorEnv(thunks) if asynchronous else gym.vector.SyncVectorEnv(thunks)


def simulate_equal_episodes_per_task(envs_eval, agent: Agent, eval_episodes_per_task: int, device):
    """Evaluate the deterministic (mean-action) policy with equal episodes per task."""
    num_tasks = envs_eval.num_envs
    returns_by_task = [[] for _ in range(num_tasks)]
    success_by_task = [[] for _ in range(num_tasks)]
    counts = np.zeros(num_tasks, dtype=np.int32)

    obs, _ = envs_eval.reset()
    while np.any(counts < eval_episodes_per_task):
        with torch.no_grad():
            actions = agent.get_action(torch.as_tensor(obs, dtype=torch.float32).to(device), sample=False)
        obs, _, _, _, infos = envs_eval.step(actions.cpu().numpy())

        if "final_info" in infos:
            for i, finfo in enumerate(infos["final_info"]):
                if finfo and "episode" in finfo and counts[i] < eval_episodes_per_task:
                    returns_by_task[i].append(episode_scalar(finfo["episode"]["r"]))
                    success_by_task[i].append(episode_scalar(finfo.get("is_success", finfo.get("success", 0.0))))
                    counts[i] += 1

    per_task = []
    for i in range(num_tasks):
        r = np.asarray(returns_by_task[i], dtype=np.float64)
        s = np.asarray(success_by_task[i], dtype=np.float64)
        per_task.append((float(r.mean()), float(r.std()), float(s.mean()), float(s.std())))
    return per_task


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    env_name = args.env_name if args.env_name else "_".join(args.env_ids)
    run_name = f"{env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.num_evals is not None:
        args.eval_freq = max(args.num_iterations // args.num_evals, 1)

    if args.seed is None:
        args.seed = args.run_id if args.run_id else np.random.randint(2**31 - 1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Output path
    args.output_dir = f"{args.output_rootdir}/{env_name}/ppo_sepnn/{args.output_subdir}"
    if args.run_id is not None:
        args.output_dir += f"/run_{args.run_id}"
    else:
        args.output_dir += f"/run_{get_latest_run_id(log_path=args.output_dir, log_name='run') + 1}"
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

    num_tasks = len(args.env_ids)

    envs = make_multitask_train_vec_env(
        env_ids=args.env_ids,
        num_envs=args.num_envs,
        capture_video=args.capture_video,
        run_name=run_name,
        task_probs_init=args.task_probs_init,
        seed=args.seed,
        asynchronous=args.asynchronous,
    )
    envs_eval = make_eval_vec_env(args.env_ids, asynchronous=args.asynchronous)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, num_tasks=num_tasks, hidden_dim=args.hidden_dim, linear=args.linear).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"{num_tasks} task networks, {sum(p.numel() for p in agent.parameters())} parameters total")

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    gap_estimator = ReturnGapEstimator(
        num_tasks=num_tasks,
        return_window=args.dro_return_window,
        success_window=args.success_window,
        return_ref_init=args.return_ref_init,
        return_rand=args.return_rand,
        rand_window=args.dro_rand_window,
        ref_success_threshold=args.ref_success_threshold,
        use_success_as_return=args.dro_success_ref,
    )

    task_probs = envs.get_task_probs()
    # Only used by drats_reweight, where sampling stays uniform and the loss is reweighted.
    task_weights = np.ones(num_tasks) / num_tasks
    # PLR's prioritization signal, refreshed once per iteration from the GAE advantages.
    learning_potential = np.zeros(num_tasks)

    global_step = 0
    update_count = 0
    logs = defaultdict(list)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

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

            # The env samples a new task on autoreset, so task_id must be read
            # from final_info rather than from the live env.
            if "final_info" in infos:
                for finfo in infos["final_info"]:
                    if finfo and "episode" in finfo:
                        gap_estimator.record_episode(
                            task_id=int(finfo["task_id"]),
                            episodic_return=episode_scalar(finfo["episode"]["r"]),
                            success=episode_scalar(finfo.get("is_success", finfo.get("success", 0.0))),
                        )

            # --------------------------------------------------------------
            # Task distribution update
            # --------------------------------------------------------------
            if global_step % args.dro_num_steps == 0:
                gaps = gap_estimator.normalized_gaps()
                success_rates = gap_estimator.success_rates()

                if args.task_sampling_algo == "uniform":
                    pass
                elif args.task_sampling_algo in DRATS_ALIASES:
                    task_probs = mirror_ascent_kl_update(
                        q=task_probs,
                        gap=gaps,
                        eta=args.dro_eta,
                        step_size=args.dro_step_size,
                        eps=args.dro_eps,
                        p0=None,  # uniform by default
                    )
                elif args.task_sampling_algo in REWEIGHT_ALIASES:
                    task_weights = mirror_ascent_kl_update(
                        q=task_weights,
                        gap=gaps,
                        eta=args.dro_eta,
                        step_size=args.dro_step_size,
                        eps=args.dro_eps,
                        p0=None,
                    )
                elif args.task_sampling_algo == "learning_progress":
                    task_probs = learning_progress_update(
                        q=task_probs,
                        learning_progress=gap_estimator.learning_progress(),
                        success_rates=success_rates,
                        eta=args.dro_eta,
                        step_size=args.dro_step_size,
                        eps=args.dro_eps,
                        p0=None,
                        success_threshold=args.success_threshold,
                    )
                elif args.task_sampling_algo == "learning_potential":
                    task_probs = learning_progress_update(
                        q=task_probs,
                        learning_progress=learning_potential,
                        success_rates=success_rates,
                        eta=args.dro_eta,
                        step_size=args.dro_step_size,
                        eps=args.dro_eps,
                        p0=None,
                        success_threshold=args.success_threshold,
                    )
                elif args.task_sampling_algo == "hard_first":
                    task_probs = hard_first_curriculum_update(
                        success_rates=success_rates,
                        eps=args.dro_eps,
                        success_threshold=args.success_threshold,
                    )
                elif args.task_sampling_algo == "easy_first":
                    task_probs = easy_first_curriculum_update(
                        success_rates=success_rates,
                        eps=args.dro_eps,
                        success_threshold=args.success_threshold,
                    )
                else:
                    raise ValueError(f"unknown task_sampling_algo: {args.task_sampling_algo}")

                envs.set_task_probs(task_probs)
                gap_estimator.commit()
                if args.dro_reset_returns:
                    gap_estimator.clear_returns()

        # bootstrap value if not done
        with torch.no_grad():
            # NOTE: ppo_continuous_action_dro.py reuses values[-1] here instead of
            # evaluating V(s_T). This is the standard CleanRL bootstrap.
            next_value = agent.get_value(next_obs).reshape(1, -1)
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_task_ids = b_obs[:, :num_tasks].argmax(dim=1)

        # PLR's learning potential: mean |GAE advantage| per task, before normalization.
        for t in range(num_tasks):
            mask = b_task_ids == t
            if mask.any():
                learning_potential[t] = float(b_advantages[mask].abs().mean().item())

        # Per-task advantage normalization: normalizing globally biases learning
        # toward tasks with larger-magnitude returns.
        if args.norm_adv_per_task:
            for t in range(num_tasks):
                mask = b_task_ids == t
                if mask.sum() > 1:
                    mb = b_advantages[mask]
                    b_advantages[mask] = (mb - mb.mean()) / (mb.std() + 1e-8)

        if args.task_sampling_algo in REWEIGHT_ALIASES:
            # num_tasks * w has mean 1 when the batch is task-balanced, so
            # uniform weights recover the unweighted losses exactly.
            b_weights = torch.as_tensor(task_weights, dtype=torch.float32, device=device)[b_task_ids] * num_tasks
        else:
            b_weights = torch.ones(args.batch_size, device=device)

        # PPO update
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv and not args.norm_adv_per_task:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_weights = b_weights[mb_inds]

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_weights).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * (torch.max(v_loss_unclipped, v_loss_clipped) * mb_weights).mean()
                else:
                    v_loss = 0.5 * (((newvalue - b_returns[mb_inds]) ** 2) * mb_weights).mean()

                entropy_loss = (entropy * mb_weights).mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                if args.clip_grad_per_task:
                    for t in range(num_tasks):
                        nn.utils.clip_grad_norm_(agent.task_parameters(t), args.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_count += 1

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        if iteration % args.eval_freq == 0:
            print(f"Eval num_timesteps={global_step}")
            print(f"Task probs: {np.round(task_probs, 4)}")
            print(f"Gaps: {np.round(gap_estimator.normalized_gaps(), 4)}")

            logs["timestep"].append(global_step)
            logs["update"].append(update_count)

            per_task_stats = simulate_equal_episodes_per_task(envs_eval, agent, args.eval_episodes, device)

            return_all_tasks, success_all_tasks = [], []
            for j, (return_avg, return_std, success_avg, success_std) in enumerate(per_task_stats):
                return_all_tasks.append(return_avg)
                success_all_tasks.append(success_avg)

                print(f"Task {j}: {args.env_ids[j]}")
                print(f"episode_return={return_avg:.2f} +/- {return_std:.2f}")
                print(f"episode_success={success_avg:.2f} +/- {success_std:.2f}")
                print()

                logs[f"task_probs_{j}"].append(float(task_probs[j]))
                logs[f"task_weights_{j}"].append(float(task_weights[j]))
                logs[f"return_{j}"].append(return_avg)
                logs[f"success_rate_{j}"].append(success_avg)

            for key, value in gap_estimator.log_dict().items():
                logs[key].append(value)

            print("Average over all tasks:")
            print(f"episode_return={float(np.mean(return_all_tasks)):.2f}")
            print(f"episode_success={float(np.mean(success_all_tasks)):.2f}")
            print()

            logs["return"].append(float(np.mean(return_all_tasks)))
            logs["success_rate"].append(float(np.mean(success_all_tasks)))
            logs["worst_task_return"].append(float(np.min(return_all_tasks)))

            np.savez(f"{args.output_dir}/evaluations.npz", **logs)

    if args.save_model:
        model_path = f"{args.output_dir}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    envs_eval.close()
