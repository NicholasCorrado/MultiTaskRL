import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import custom_envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from stable_baselines3.common.utils import get_latest_run_id
from torch.distributions.categorical import Categorical

from env_wrappers import OneHotTaskWrapper, MultiTaskEnvWrapper
from task_distribution_updates import easy_first_curriculum_update, mirror_ascent_kl_update, learning_progress_update, \
    hard_first_curriculum_update
from task_samplers import MultiTaskSampler, EasyFirstTaskSampler, HardFirstTaskSampler, DROTaskSampler


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
    upload_model: bool = False
    hf_entity: str = ""

    run_id: int = None
    asynchronous: bool = False  # if True, uses AsyncVectorEnv for training multitask wrappers

    # Output
    output_rootdir: str = "results"
    output_subdir: str = ""

    # Evaluation
    eval_freq: int = 100
    num_evals: int = None
    eval_episodes: int = 100

    # Multitask + DRO
    task_sampling_algo: str = 'uniform'
    # init_task_probs: List[float] = None
    dro_success_ref: bool = False
    task_probs_init: List[float] = None
    dro_num_steps: int = 256
    dro_eps: float = 0.05 # minimum task probability
    dro_eta: float = 8.0 # controls sharpness of task distribution. Larger = sharper
    dro_step_size: float = 0.1 # don't change this

    # Algorithm specific arguments
    env_ids: List[str] = field(default_factory=lambda: [f"GridWorld{i}-v0" for i in range(1, 4 + 1)])
    total_timesteps: int =  10000000
    learning_rate: float = 3e-3
    num_envs: int = 1
    num_steps: int = 256
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 1
    update_epochs: int = 1
    norm_adv: bool = False
    clip_coef: float = 9999999
    clip_vloss: bool = True
    ent_coef: float = 0.00
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    linear: bool = True

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, linear=True):
        super().__init__()
        in_dim = int(np.array(envs.single_observation_space.shape).prod())
        out_dim = envs.single_action_space.n

        if linear:
            self.critic = nn.Sequential(layer_init(nn.Linear(in_dim, 1), std=0.01))
            self.actor = nn.Sequential(layer_init(nn.Linear(in_dim, out_dim), std=0.01))
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(in_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=0.01),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear(in_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, out_dim), std=0.01),
            )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action(self, x, sample=True):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.sample()

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
                        success_by_task[i].append(float(finfo.get("is_success", 0.0)))
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

def make_multitask_train_vec_env(env_ids, num_envs, seed=None, asynchronous=False):
    num_tasks = len(env_ids)
    init_task_probs = np.ones(num_tasks) / num_tasks

    def slot_thunk(slot_idx):
        def _thunk():
            sampler = MultiTaskSampler(init_task_probs)  # <-- important

            task_envs = []
            for task_id, env_id in enumerate(env_ids):
                env = gym.make(env_id)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env = OneHotTaskWrapper(env, task_id=task_id, num_tasks=num_tasks)
                task_envs.append(env)

            return MultiTaskEnvWrapper(task_envs, sampler=sampler, seed=None if seed is None else seed + slot_idx)
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


def make_eval_vec_env(env_ids, asynchronous=False):
    num_tasks = len(env_ids)

    def thunk(task_id):
        def _thunk():
            env = gym.make(env_ids[task_id])
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = OneHotTaskWrapper(env, task_id=task_id, num_tasks=num_tasks)
            return env
        return _thunk

    thunks = [thunk(t) for t in range(num_tasks)]
    return gym.vector.AsyncVectorEnv(thunks) if asynchronous else gym.vector.SyncVectorEnv(thunks)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # env_name = "_".join(args.env_ids)
    env_name = ""
    run_name = f"{env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.num_evals is not None:
        args.eval_freq = args.num_iterations // args.num_evals

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

    num_tasks = len(args.env_ids)

    envs = make_multitask_train_vec_env(
        env_ids=args.env_ids,
        num_envs=args.num_envs,
        seed=args.seed,
        asynchronous=args.asynchronous,
    )

    # Deterministic eval vec env: one env per task
    envs_eval = make_eval_vec_env(
        env_ids=args.env_ids,
        asynchronous=args.asynchronous,
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, linear=args.linear).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup (unchanged)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    objective_weights = torch.ones((args.num_steps, args.num_envs)).to(device)

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

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            objective_weights[step] = current_task_objective_weights[current_task_id]

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
                        task_success_buffer[current_task_id].append(float(finfo.get("is_success", 0.0)))
                        current_task_id = envs.get_task_id()

            if global_step % args.dro_num_steps == 0:
                training_returns_avg = np.array([np.mean(training_returns[i]) if len(training_returns[i]) > 0 else 0 for i in range(num_tasks)])
                return_ref = np.ones(num_tasks)
                # return_ref = np.array([0.95, 0.93, 0.91, 0.89])

                return_gap = np.clip(return_ref - training_returns_avg, 0, np.inf) / return_ref
                return_slope = np.abs(training_returns_avg - training_returns_avg_prev) if training_returns_avg_prev is not None else np.zeros(num_tasks)


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

        # PPO update (unchanged)
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                if args.task_sampling_algo == 'dro_reweight':
                    pg_loss = (torch.max(pg_loss1, pg_loss2) * objective_weights[mb_inds]).sum()
                else:
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    if args.task_sampling_algo == 'dro_reweight':
                        v_loss = 0.5 * (torch.max(v_loss_unclipped, v_loss_clipped) * objective_weights[mb_inds]).sum()
                    else:
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    if args.task_sampling_algo == 'dro_reweight':
                        v_loss = 0.5 * (((newvalue - b_returns[mb_inds]) * objective_weights[mb_inds]) ** 2).sum()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                # print(grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_count += 1

        # Eval (refactored only: deterministic, equal episodes per task)
        if iteration % args.eval_freq == 0:
            # optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

            print(f"Eval num_timesteps={global_step}")
            p_now = envs.get_task_probs()
            print(f"Task probs: {p_now}")

            logs["timestep"].append(global_step)
            logs["update"].append(update_count)

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

                logs[f"task_probs_{j}"].append(current_task_distribution[j])
                logs[f"task_weights_{j}"].append(current_task_objective_weights[j])
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