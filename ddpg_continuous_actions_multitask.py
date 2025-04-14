# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import copy
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import gymnasium as gym
# import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import yaml
# from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import get_latest_run_id

from utils import simulate


def simulate(env, actor, eval_episodes, eval_steps=np.inf, device='cpu'):
    logs = defaultdict(list)
    step = 0
    for episode_i in range(eval_episodes):
        logs_episode = defaultdict(list)

        obs, _ = env.reset()
        done = False

        while not done:

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                actions = actor.get_action(torch.Tensor(obs).to(device), sample=False)
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
        try:
            logs['successes'].append(infos['is_success'])
        except:
            logs['successes'].append(False)

    return_avg = np.mean(logs['returns'])
    return_std = np.std(logs['returns'])
    success_avg = np.mean(logs['successes'])
    success_std = np.std(logs['successes'])
    return return_avg, return_std, success_avg, success_std

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, hidden_dims=None):
        super().__init__()

        # Default to [256, 256] if hidden_dims is None
        if hidden_dims is None:
            hidden_dims = [256, 256]

        input_dim = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)

        # Build Q network
        q_layers = []

        # First layer
        q_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        q_layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            q_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            q_layers.append(nn.ReLU())

        # Output layer
        q_layers.append(nn.Linear(hidden_dims[-1], 1))

        self.q_net = nn.Sequential(*q_layers)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.q_net(x)


class Actor(nn.Module):
    def __init__(self, env, hidden_dims=None):
        super().__init__()

        # Default to [256, 256] if hidden_dims is None
        if hidden_dims is None:
            hidden_dims = [256, 256]

        input_dim = np.array(env.single_observation_space.shape).prod()
        output_dim = np.prod(env.single_action_space.shape)

        # Build policy network
        policy_layers = []

        # First layer
        policy_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        policy_layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            policy_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            policy_layers.append(nn.ReLU())

        # Output layer
        policy_layers.append(nn.Linear(hidden_dims[-1], output_dim))
        policy_layers.append(nn.Tanh())

        self.policy_net = nn.Sequential(*policy_layers)

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.policy_net(x)
        return x * self.action_scale + self.action_bias

    def get_action(self, x, sample=True):
        # For compatibility with simulate function from PPO
        action = self.forward(x)
        # Note that we don't add sampling noise here during evaluation
        return action


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
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
    save_policy: bool = False

    # Logging
    output_rootdir: str = 'results'
    output_subdir: str = ''
    run_id: int = None
    seed: int = None

    # Evaluation
    num_evals: int = 40
    eval_freq: int = None
    eval_episodes: int = 20

    # Algorithm specific arguments
    env_id: str = "InvertedPendulum"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    def __post_init__(self):

        # Seeding
        if self.run_id:
            self.seed = self.run_id
        elif self.seed is None:
            self.seed = np.random.randint(2 ** 32 - 1)

        # Output path setup
        self.output_dir = f"{self.output_rootdir}/{self.env_id}/ddpg/{self.output_subdir}"
        if self.run_id is not None:
            self.output_dir += f"/run_{self.run_id}"
        else:
            run_id = get_latest_run_id(log_path=self.output_dir, log_name='run_') + 1
            self.output_dir += f"/run_{run_id}"

        if self.eval_freq is None:
            self.eval_freq = max(self.total_timesteps // self.num_evals, 1)

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.output_dir,
            monitor_gym=True,
            save_code=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, "") for i in range(args.num_envs)]
    )
    envs_eval = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, "") for i in range(1)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    # Logging
    logs = defaultdict(list)
    eval_count = 0
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        # Evaluation
        if global_step % args.eval_freq == 0:
            eval_count += 1

            return_avg, return_std, success_avg, success_std = simulate(
                env=envs_eval,
                actor=actor,
                eval_episodes=args.eval_episodes
            )

            print(
                f"Eval num_timesteps={global_step}, "
                f"episode_return={return_avg:.2f} +/- {return_std:.2f}\n"
                f"episode_success={success_avg:.2f} +/- {success_std:.2f}\n"
            )

            # Log metrics
            logs['timestep'].append(global_step)
            logs['return'].append(return_avg)
            logs['success_rate'].append(success_avg)

            if global_step > args.learning_starts:
                # Log DDPG specific metrics
                logs['ddpg/q_loss'].append(qf1_loss.item())
                logs['ddpg/q_values'].append(qf1_a_values.mean().item())
                logs['ddpg/actor_loss'].append(actor_loss.item() if 'actor_loss' in locals() else 0.0)

            # Calculate steps per second
            sps = int(global_step / (time.time() - start_time))
            logs['sps'].append(sps)

            # Save logs
            np.savez(
                f'{args.output_dir}/evaluations.npz',
                **logs,
            )

            # Save policy if requested
            if args.save_policy:
                torch.save(actor, f"{args.output_dir}/policy_{eval_count}.pt")

            # Log to wandb if tracking
            if args.track:
                log_wandb = {}
                for key, value in logs.items():
                    log_wandb[key] = value[-1]
                wandb.log(log_wandb)

    if args.save_model:
        model_path = f"{args.output_dir}/policy.pt"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")

    envs.close()