# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import gymnasium as gym
import custom_envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from stable_baselines3.common.utils import get_latest_run_id
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils import simulate

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    run_id: int = None
    """Results will be saved to {output_dir}/run_{run_id} and seed will be set to run_id"""

    # Output
    output_rootdir: str = "results"
    """Top level directory to which results will be saved"""
    output_subdir: str = ""
    """Results are saved to {output_rootdir}/ppo/{output_subdir}/run_{run_id}"""

    # Evaluation
    eval_freq: int = 100
    """Evaluate policy every eval_freq updates"""
    eval_episodes: int = 100
    """Number of trajectories to collect during each evaluation"""

    task_probs_init: List[float] = None
    dro: bool = False
    dro_num_steps: int = 8192
    dro_learning_rate: float = 0.1
    dro_eps: float = 0.1
    dro_success_ref: bool = True
    dro_easy_first: bool = False
    dro_td: bool = False
    dro_disc: float = 0.9
    dro_disc_on: bool = False
    """Use the improvement between two DRO sampling as reference"""

    linear: bool = False
    """Use a linear actor/critic network"""

    # Algorithm specific arguments
    env_ids: List[str] = field(default_factory=lambda: [f"Goal2D{i}-v0" for i in [1,4]])
    # env_ids: List[str] = field(default_factory=lambda: [f"BanditEasy-v0", "BanditHard-v0"])
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""



def simulate(env, actor, eval_episodes, eval_steps=np.inf):
    logs = defaultdict(list)
    step = 0
    for episode_i in range(eval_episodes):
        logs_episode = defaultdict(list)

        obs, _ = env.reset()
        done = False

        while not done:

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                actions = actor.get_action(torch.Tensor(obs).to('cpu'), sample=False)
                actions = actions.cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            done = np.logical_or(terminateds, truncateds)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            logs_episode['rewards'].append(rewards)

            step += 1

            if step >= eval_steps:
                break
        if step >= eval_steps:
            break

        logs['returns'].append(np.sum(logs_episode['rewards']))
        logs['successes'].append(infos['final_info'][0]['is_success'])

    return_avg = np.mean(logs['returns'])
    return_std = np.std(logs['returns'])
    success_avg = np.mean(logs['successes'])
    success_std = np.std(logs['successes'])
    return return_avg, return_std, success_avg, success_std

def make_env(env_id, idx, capture_video, run_name, gamma):

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, linear=False):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action(self, x, sample=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        return action

def exponentiated_gradient_ascent_step(w, returns, returns_ref, previous_return_avg, task_probs, learning_rate=1.0, eps=0.1):
    # Use s_t - s_{t-1} instead of s_ref - s_t
    if args.dro_td:
        diff = np.clip(returns_ref - previous_return_avg, 0, np.inf)
    else:
        diff = np.clip(returns_ref - returns, 0, np.inf)

    # Prioritize easy tasks (don't use it!)
    if args.dro_easy_first:
        diff *= -1.0

    # Add discount factor to early results
    if args.dro_disc_on:
        w_new = (w ** args.dro_disc) * np.exp(learning_rate * diff)
    else:
        # Exponentiated gradient update
        w_new = w * np.exp(learning_rate * diff)

    # Normalize to ensure weights sum to 1
    w_new = w_new / w_new.sum()

    # Smoothing to prevent weights form getting too close to 0
    w_uniform = 1/len(w_new) * np.ones(len(w_new))
    w_new = (1 - eps) * w_new + eps * w_uniform

    return w_new

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    env_name = ""
    for env_id in args.env_ids:
        env_name += env_id + "_"
    env_name = env_name[:-1]

    run_name = f"{env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Seeding
    if args.seed is None:
        if args.run_id:
            args.seed = args.run_id
        else:
            args.seed = np.random.randint(2 ** 32 - 1)

            # TRY NOT TO MODIFY: seeding
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

    # Dump training config to save dir
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)


    # wandb
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
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    # env setup
    num_tasks = len(args.env_ids)
    envs_list = []
    envs_eval_list = []
    for task_id in range(num_tasks):
        print(args.env_ids[task_id])
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_ids[task_id], i, args.capture_video, run_name, gamma = args.gamma) for i in range(args.num_envs)],
        )
        envs_eval = gym.vector.SyncVectorEnv(
            [make_env(args.env_ids[task_id], i, args.capture_video, run_name, gamma = args.gamma) for i in range(1)],
        )

        envs_list.append(envs)
        envs_eval_list.append(envs_eval)

    if args.task_probs_init:
        task_probs = np.array(args.task_probs_init)
    else:
        task_probs = np.ones(num_tasks) / num_tasks

    task_weights = np.ones(num_tasks) / num_tasks

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only discrete action space is supported"

    agent = Agent(envs, linear=args.linear).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    update_count = 0
    start_time = time.time()
    logs = defaultdict(lambda: [])

    training_returns = [[] for i in range(num_tasks)]

    next_obs_list = []
    next_done_list = []
    for task_id in range(num_tasks):
        envs = envs_list[task_id]
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        next_obs_list.append(next_obs)
        next_done_list.append(next_done)

    task_id = np.random.choice(np.arange(num_tasks), p=task_probs)
    envs = envs_list[task_id]

    next_obs = next_obs_list[task_id]
    next_done = next_done_list[task_id]

    # Buffer for the last data episode
    previous_return_avg = np.zeros(num_tasks)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done, terminations = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device), torch.Tensor(terminations)

            next_obs_list[task_id] = next_obs
            next_done_list[task_id] = next_done

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        # training_returns[task_id].append(np.mean(info["episode"]["r"]))
                        if args.dro_success_ref:
                            training_returns[task_id].append(np.mean(info['is_success']))
                        else:
                            training_returns[task_id].append(np.mean(info["episode"]["r"]))

                        task_id = np.random.choice(np.arange(num_tasks), p=task_probs)
                        envs = envs_list[task_id]
                        next_obs = next_obs_list[task_id]
                        next_done = next_done_list[task_id]

            # Update task sampling probabilities
            if args.dro and global_step % args.dro_num_steps == 0:
                # Calculate the performance of this episode
                training_returns_avg = np.array([np.mean(training_returns[i]) for i in range(num_tasks)])
                training_returns_avg = np.nan_to_num(training_returns_avg)

                # The return reference
                returns_ref = np.ones(num_tasks)

                # Update the sampling probability
                task_probs = exponentiated_gradient_ascent_step(task_probs, training_returns_avg, returns_ref, previous_return_avg, task_probs,
                                                                learning_rate=args.dro_learning_rate, eps=args.dro_eps)

                # Update the previous episode's performance
                previous_return_avg = training_returns_avg
                training_returns = [[] for i in range(num_tasks)]

        # bootstrap value if not done
        with torch.no_grad():
            next_value = values[-1].reshape(1, -1)
            # next_value = agent.get_value(next_obs).reshape(1, -1)
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

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
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

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        update_count += 1

        # print(task_probs)

        if iteration % args.eval_freq == 0:
            print(f"Eval num_timesteps={global_step}")
            print(f'Task probs: {task_probs}')
            logs['timestep'].append(global_step)
            logs['update'].append(update_count)

            return_all_tasks = []
            std_all_tasks = []
            success_all_tasks = []

            for j in range(num_tasks):
                envs_eval = envs_eval_list[j]
                return_avg, return_std, success_avg, success_std = simulate(env=envs_eval, actor=agent, eval_episodes=args.eval_episodes)

                return_all_tasks.append(return_avg)
                success_all_tasks.append(success_avg)

                print(f"Task {j}: {args.env_ids[j]}")
                print(f"episode_return={return_avg:.2f} +/- {return_std:.2f}")
                print(f"episode_success={success_avg:.2f} +/- {success_std:.2f}")
                print()

                logs[f'task_probs_{j}'].append(task_probs[j])
                logs[f'return_{j}'].append(return_avg)
                logs[f'success_rate_{j}'].append(success_avg)


            return_all_tasks_avg = np.mean(return_all_tasks)
            return_all_tasks_std = np.sqrt(np.sum(np.array(std_all_tasks)**2))

            success_all_tasks_avg = np.mean(success_all_tasks)
            # return_all_tasks_std = np.sqrt(np.sum(np.array(std_all_tasks)**2))

            print(f"Average over all tasks:")
            print(f"episode_return={return_all_tasks_avg:.2f} +/- {return_all_tasks_std:.2f}")
            print(f"episode_success={success_all_tasks_avg:.2f} +/- ")
            print()

            logs['return'].append(return_all_tasks_avg)
            logs['success_rate'].append(success_all_tasks_avg)

            np.savez(
                f'{args.output_dir}/evaluations.npz',
                **logs,
            )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    # writer.close()