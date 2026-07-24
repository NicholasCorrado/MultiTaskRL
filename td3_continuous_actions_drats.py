"""Multi-task TD3 with Distributionally Robust Adaptive Task Sampling (DRATS).

Identical to `ddpg_continuous_actions_drats.py` except for the learner: twin
critics, target policy smoothing, and delayed policy/target updates. Everything
about how the task distribution q is estimated and updated is shared with the
DDPG and PPO scripts (see `return_gap.py` and `task_distribution_updates.py`),
so the three algorithms differ only in the RL update.

Baselines available through --task_sampling_algo:
    uniform             fixed uniform task distribution
    drats               DRATS (alias: dro)
    drats_reweight      DRO on the *objective* instead of the sampler: tasks are
                        sampled uniformly and the critic/actor losses are
                        reweighted by q (alias: dro_reweight)
    learning_progress   ALP-GMM / RIAC: |J_hat_t - J_hat_{t-1}|
    learning_potential  PLR adapted to off-policy: mean |TD error| per task
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
import torch.nn.functional as F
import torch.optim as optim
import tyro
import yaml
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import get_latest_run_id

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
    asynchronous: bool = False
    """if True, uses AsyncVectorEnv for the training envs"""

    # Output
    output_rootdir: str = "results"
    output_subdir: str = ""
    env_name: str = None
    """name of the benchmark used in the output path; defaults to the joined env ids"""

    # Evaluation
    num_evals: int = 100
    eval_freq: int = None
    """evaluate every eval_freq environment steps; overridden by num_evals if unset"""
    eval_episodes: int = 20
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
    """random-policy return used to normalize gaps; None = estimate it from the warm-up phase"""
    dro_rand_window: int = 0
    """extra episodes per task to fold into the J_rand estimate; the uniform-random
    warm-up phase already supplies it, so this is normally 0"""
    ref_success_threshold: float = 0.5
    """success rate above which J_ref switches to the max observed return"""

    # Algorithm specific arguments
    env_ids: List[str] = field(default_factory=lambda: ["PointMaze_UMaze-v3", "PointMaze_Medium-v3", "PointMaze_Large-v3"])
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    exploration_noise: float = 0.1
    learning_starts: int = 25_000
    policy_frequency: int = 2
    """delayed policy update: update the actor and targets every policy_frequency steps"""
    policy_noise: float = 0.2
    """stddev of the target policy smoothing noise"""
    noise_clip: float = 0.5
    """clip range of the target policy smoothing noise"""
    num_envs: int = 1
    """number of parallel training envs; each collects one transition per global step"""


# ----------------------------
# Networks
# ----------------------------

class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


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

    # Vector-level get/set that works for both Sync and Async via envs.call.
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


def simulate_equal_episodes_per_task(envs_eval, actor: Actor, eval_episodes_per_task: int, device):
    """Evaluate the deterministic (noise-free) policy with equal episodes per task."""
    num_tasks = envs_eval.num_envs
    returns_by_task = [[] for _ in range(num_tasks)]
    success_by_task = [[] for _ in range(num_tasks)]
    counts = np.zeros(num_tasks, dtype=np.int32)

    obs, _ = envs_eval.reset()
    while np.any(counts < eval_episodes_per_task):
        with torch.no_grad():
            actions = actor(torch.as_tensor(obs, dtype=torch.float32).to(device)).cpu().numpy()
        actions = actions.clip(envs_eval.single_action_space.low, envs_eval.single_action_space.high)

        obs, _, _, _, infos = envs_eval.step(actions)

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

    env_name = args.env_name if args.env_name else "_".join(args.env_ids)
    run_name = f"{env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.seed is None:
        args.seed = args.run_id if args.run_id else np.random.randint(2**31 - 1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.eval_freq is None:
        args.eval_freq = max(args.total_timesteps // args.num_evals, 1)

    # Output path
    args.output_dir = f"{args.output_rootdir}/{env_name}/td3/{args.output_subdir}"
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
    envs.single_action_space.seed(args.seed)

    actor = Actor(envs).to(device)
    target_actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

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
    # Off-policy stand-in for PLR's L1 value loss: mean |TD error| per task.
    td_error_sum = np.zeros(num_tasks)
    td_error_count = np.zeros(num_tasks)

    logs = defaultdict(list)
    update_count = 0
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.as_tensor(obs, dtype=torch.float32).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Record finished episodes. The env samples a new task on autoreset, so
        # task_id must be read from final_info rather than from the live env.
        if "final_info" in infos:
            for finfo in infos["final_info"]:
                if finfo and "episode" in finfo:
                    gap_estimator.record_episode(
                        task_id=int(finfo["task_id"]),
                        episodic_return=episode_scalar(finfo["episode"]["r"]),
                        success=episode_scalar(finfo.get("is_success", finfo.get("success", 0.0))),
                        random_policy=global_step < args.learning_starts,
                    )

        # Save to the replay buffer, substituting the true terminal observation
        # for the autoreset observation (which belongs to a different episode,
        # and usually to a different task).
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, done in enumerate(np.logical_or(terminations, truncations)):
                if done and infos["final_observation"][idx] is not None:
                    real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # J_rand is whatever the uniform-random warm-up policy achieved.
        if global_step == args.learning_starts:
            print(f"J_rand estimate: {gap_estimator.finalize_random_returns()}")

        # ------------------------------------------------------------------
        # Task distribution update
        # ------------------------------------------------------------------
        if global_step >= args.learning_starts and global_step % args.dro_num_steps == 0:
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
                td_error_avg = np.where(td_error_count > 0, td_error_sum / np.maximum(td_error_count, 1), 0.0)
                task_probs = learning_progress_update(
                    q=task_probs,
                    learning_progress=td_error_avg,
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
            td_error_sum[:] = 0.0
            td_error_count[:] = 0.0

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            # The one-hot prefix of the observation identifies each sample's task.
            batch_task_ids = data.observations[:, :num_tasks].argmax(dim=1)
            if args.task_sampling_algo in REWEIGHT_ALIASES:
                # num_tasks * w has mean 1 when the batch is task-balanced, so
                # uniform weights recover the unweighted losses exactly.
                sample_weights = (
                    torch.as_tensor(task_weights, dtype=torch.float32, device=device)[batch_task_ids] * num_tasks
                )
            else:
                sample_weights = None

            with torch.no_grad():
                # target policy smoothing
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale
                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    torch.as_tensor(envs.single_action_space.low, dtype=torch.float32, device=device),
                    torch.as_tensor(envs.single_action_space.high, dtype=torch.float32, device=device),
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            td_error1 = qf1_a_values - next_q_value
            td_error2 = qf2_a_values - next_q_value
            if sample_weights is None:
                qf_loss = (td_error1**2).mean() + (td_error2**2).mean()
            else:
                qf_loss = (sample_weights * td_error1**2).mean() + (sample_weights * td_error2**2).mean()

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if args.task_sampling_algo == "learning_potential":
                with torch.no_grad():
                    abs_td = td_error1.abs().detach().cpu().numpy()
                    ids = batch_task_ids.cpu().numpy()
                    np.add.at(td_error_sum, ids, abs_td)
                    np.add.at(td_error_count, ids, 1)

            if global_step % args.policy_frequency == 0:
                q_pi = qf1(data.observations, actor(data.observations)).view(-1)
                actor_loss = -q_pi.mean() if sample_weights is None else -(sample_weights * q_pi).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target networks
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            update_count += 1

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        if global_step > args.learning_starts and global_step % args.eval_freq == 0:
            print(f"Eval num_timesteps={global_step}  SPS={int(global_step / (time.time() - start_time))}")
            print(f"Task probs: {np.round(task_probs, 4)}")
            print(f"Gaps: {np.round(gap_estimator.normalized_gaps(), 4)}")

            logs["timestep"].append(global_step)
            logs["update"].append(update_count)

            per_task_stats = simulate_equal_episodes_per_task(envs_eval, actor, args.eval_episodes, device)

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
        torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    envs_eval.close()
