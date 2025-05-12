# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
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
import torch.nn.functional as F
import torch.optim as optim
import tyro
import yaml
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


from stable_baselines3.common.utils import get_latest_run_id

from utils import simulate, simulate_ddpg


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
    """Results are saved to {output_rootdir}/ddpg/{output_subdir}/run_{run_id}"""

    # Evaluation
    num_evals: int = 100
    eval_freq: int = 100
    """Evaluate policy every eval_freq updates"""
    eval_episodes: int = 20
    """Number of trajectories to collect during each evaluation"""

    # DRO setting
    task_probs_init: List[float] = field(default_factory=lambda: [1/4 for i in range(4)])
    dro: bool = False
    dro_num_steps: int = 128
    dro_learning_rate: float = 1.0
    dro_eps: float = 0.01
    dro_success_ref: bool = False

    # Algorithm specific arguments
    env_ids: List[str] = field(default_factory=lambda: [f"Goal2D{i}-v0" for i in range(1, 4+1)])
    """the environment id of the Atari game"""
    num_envs: int = 1
    """the number of parallel game environments"""
    total_timesteps: int = 300000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
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



def make_env(env_id, idx, capture_video, run_name):

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def simulate_ddpg(env, actor, eval_episodes, device, exploration_noise, single_action_space, eval_steps=np.inf):
    logs = defaultdict(list)
    step = 0
    num_env = 0
    for episode_i in range(eval_episodes):
        logs_episode = defaultdict(list)

        obs, _ = env.reset()
        done = False
        Done = False

        while not Done:

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * exploration_noise)
                actions = actions.cpu().numpy().clip(single_action_space.low, single_action_space.high)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            done = np.logical_or(terminateds, truncateds)
            Done = done.all()

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


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    env_name = ""
    for env_id in args.env_ids:
        env_name += env_id + "_"
    env_name = env_name[:-1]

    run_name = f"{env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


    # Output path
    args.output_dir = f"{args.output_rootdir}/{env_name}/ddpg/{args.output_subdir}"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    args.eval_freq = max(args.total_timesteps // args.num_evals, 1)

    # env setup

    num_tasks = len(args.env_ids)
    envs_list = []
    envs_eval_list = []
    for task_id in range(num_tasks):
        print(args.env_ids[task_id])
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_ids[task_id], i, args.capture_video, run_name) for i in range(args.num_envs)],
        )
        envs_eval = gym.vector.SyncVectorEnv(
            [make_env(args.env_ids[task_id], i, args.capture_video, run_name) for i in range(1)],
        )

        envs_list.append(envs)
        envs_eval_list.append(envs_eval)

    if args.task_probs_init:
        task_probs = np.array(args.task_probs_init)
    else:
        task_probs = np.ones(num_tasks) / num_tasks

    task_weights = np.ones(num_tasks) / num_tasks

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
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    logs = defaultdict(lambda: [])

    update_count = 0

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

        next_obs_list[task_id] = next_obs
        next_done_list[task_id] = next_done

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break


        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        task_id = np.random.choice(np.arange(num_tasks), p=task_probs)
        envs = envs_list[task_id]
        next_obs = next_obs_list[task_id]
        next_done = next_done_list[task_id]

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

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            update_count += 1

            if global_step % args.eval_freq == 0:
                print(f"Eval num_timesteps={global_step}")
                print(f'Task probs: {task_probs}')
                logs['timestep'].append(global_step)
                logs['update'].append(update_count)

                return_all_tasks = []
                std_all_tasks = []
                success_all_tasks = []

                for j in range(num_tasks):
                    envs_eval = envs_eval_list[j]
                    return_avg, return_std, success_avg, success_std = simulate_ddpg(env=envs_eval, actor=actor,
                                                                                eval_episodes=args.eval_episodes,
                                                                                exploration_noise=args.exploration_noise,
                                                                                device=device, single_action_space=envs.single_action_space
                                                                                )

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
                return_all_tasks_std = np.sqrt(np.sum(np.array(std_all_tasks) ** 2))

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
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")
        # from cleanrl_utils.evals.ddpg_eval import evaluate

        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     args.env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=(Actor, QNetwork),
        #     device=device,
        #     exploration_noise=args.exploration_noise,
        # )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub
        #
        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "DDPG", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
