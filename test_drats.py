"""Invariant checks for the DRATS experiment scripts.

Run with:  python test_drats.py

Covers the pieces that are easy to get silently wrong and would quietly ruin a
run: the normalized return gap, the per-task network routing in
ppo_continuous_action_drats_sepnn.py, and the multi-task collection loop used by
the off-policy scripts.
"""

# ==========================================================================
# Return gap estimation and task-distribution updates
# ==========================================================================

import numpy as np

from return_gap import ReturnGapEstimator
from task_distribution_updates import mirror_ascent_kl_update, kl_project_with_floor


def approx(a, b, tol=1e-8):
    assert np.allclose(a, b, atol=tol), f"{a} != {b}"


# 1. Explicit ref/rand: gap = (ref - J) / (ref - rand)
e = ReturnGapEstimator(3, return_ref_init=100.0, return_rand=0.0)
for _ in range(5):
    e.record_episode(0, 100.0)   # solved
    e.record_episode(1, 50.0)    # halfway
    e.record_episode(2, 0.0)     # at random performance
approx(e.estimated_returns(), [100.0, 50.0, 0.0])
approx(e.reference_returns(), [100.0, 100.0, 100.0])
approx(e.normalized_gaps(), [0.0, 0.5, 1.0])
print("gap.1 ok: gap = (ref - J)/(ref - rand)")

# 2. Negative returns: J_rand comes from the first rand_window episodes.
#    With J_rand pinned to 0 instead, the denominator J_ref - J_rand goes
#    negative and every gap saturates at 0 or 1.
e = ReturnGapEstimator(3, rand_window=2)
for _ in range(2):                       # random-policy phase
    for t in range(3):
        e.record_episode(t, -10.0)
for _ in range(5):                       # trained policy
    e.record_episode(0, -2.0)
    e.record_episode(1, -6.0)
    e.record_episode(2, -10.0)
approx(e._rand(), [-10.0, -10.0, -10.0])
approx(e.reference_returns(), [-2.0, -6.0, -10.0])  # max observed
gaps = e.normalized_gaps()
approx(gaps[:2], [0.0, 0.0])   # J_hat == max observed for both
assert gaps[2] == 1.0, gaps    # ref == rand -> degenerate -> max priority
print("gap.2 ok: negative returns do not saturate the normalization")

# 3. A task with no data at all is treated as unsolved (max gap).
e = ReturnGapEstimator(2, return_ref_init=10.0, return_rand=0.0)
for _ in range(5):
    e.record_episode(0, 10.0)
approx(e.normalized_gaps(), [0.0, 1.0])
approx(e.episode_counts(), [5, 0])
print("gap.3 ok: unseen task gets the largest gap")

# 4. Reference latches from the r_max*H_max bound to the max observed return
#    once the success rate clears the threshold.
e = ReturnGapEstimator(2, return_ref_init=1000.0, return_rand=0.0, ref_success_threshold=0.5)
for _ in range(10):
    e.record_episode(0, 300.0, success=1.0)   # success rate 1.0 -> latch
    e.record_episode(1, 300.0, success=0.0)   # success rate 0.0 -> keep bound
approx(e.reference_returns(), [300.0, 1000.0])
approx(e.normalized_gaps(), [0.0, 0.7])
for _ in range(10):                            # latch must not un-latch
    e.record_episode(0, 300.0, success=0.0)
approx(e.reference_returns()[0], 300.0)
print("gap.4 ok: reference return latches to max observed and stays latched")

# 5. Success-rate mode: J_ref = 1, J_rand = 0, regardless of the return scale.
#    Episode returns must not leak into the success-rate normalization.
for kwargs in ({}, {"rand_window": 3}, {"return_rand": -50.0}):
    e = ReturnGapEstimator(2, use_success_as_return=True, **kwargs)
    for i in range(10):
        e.record_episode(0, -10.0, success=1.0 if i < 8 else 0.0)
        e.record_episode(1, -10.0, success=0.0)
    approx(e.normalized_gaps(), [0.2, 1.0])
print("gap.5 ok: success-rate gaps ignore the return scale")

# 6. Learning progress needs commit() between updates.
e = ReturnGapEstimator(2, return_ref_init=10.0, return_rand=0.0, return_window=1)
e.record_episode(0, 1.0); e.record_episode(1, 1.0)
approx(e.learning_progress(), [0.0, 0.0])
e.commit()
e.record_episode(0, 4.0); e.record_episode(1, 1.0)
approx(e.learning_progress(), [3.0, 0.0])
print("gap.6 ok: learning progress")

# 7. Mirror ascent prioritizes the largest gap and respects the floor.
q = np.ones(4) / 4
for _ in range(50):
    q = mirror_ascent_kl_update(q=q, gap=np.array([0.0, 0.1, 0.5, 1.0]), eta=16.0, step_size=0.2, eps=0.01)
assert np.argmax(q) == 3, q
assert np.all(q >= 0.01 - 1e-12), q
approx(q.sum(), 1.0)
# equal gaps -> uniform
q = np.array([0.7, 0.1, 0.1, 0.1])
for _ in range(50):
    q = mirror_ascent_kl_update(q=q, gap=np.ones(4), eta=16.0, step_size=0.2, eps=0.01)
approx(q, np.ones(4) / 4, tol=1e-4)  # contraction rate (1-step_size)^50
print("gap.7 ok: mirror ascent")

# 8. KL projection floor
q = kl_project_with_floor(np.array([0.98, 0.01, 0.005, 0.005]), eps=0.05)
assert np.all(q >= 0.05 - 1e-12), q
approx(q.sum(), 1.0)
print("gap.8 ok: kl projection floor")



# ==========================================================================
# Per-task network routing (ppo_continuous_action_drats_sepnn)
# ==========================================================================

import numpy as np
import torch

import gymnasium as gym
import custom_envs  # noqa: F401

from ppo_continuous_action_drats_sepnn import Agent, make_multitask_train_vec_env

NUM_TASKS = 3
envs = make_multitask_train_vec_env(
    env_ids=["Goal2D-v0"] * NUM_TASKS, num_envs=1, capture_video=False, run_name="t", seed=0, asynchronous=False
)
torch.manual_seed(0)
agent = Agent(envs, num_tasks=NUM_TASKS, hidden_dim=16)
obs_dim = agent.obs_dim
act_dim = agent.act_dim

# Give each task's networks visibly different weights.
with torch.no_grad():
    for t in range(NUM_TASKS):
        for p in agent.task_parameters(t):
            p.add_(float(t + 1))

# Build a batch with a shuffled mix of tasks.
rng = np.random.default_rng(0)
B = 32
task_ids = rng.integers(0, NUM_TASKS, size=B)
base_obs = torch.as_tensor(rng.normal(size=(B, obs_dim)), dtype=torch.float32)
onehot = torch.zeros(B, NUM_TASKS)
onehot[torch.arange(B), torch.as_tensor(task_ids)] = 1.0
x = torch.cat([onehot, base_obs], dim=1)

# 1. Routing: every row must equal that task's network applied to that row.
with torch.no_grad():
    values = agent.get_value(x)
    means = agent._dispatch(base_obs, torch.as_tensor(task_ids), agent.actor_means, act_dim)
for i in range(B):
    t = int(task_ids[i])
    with torch.no_grad():
        ref_v = agent.critics[t](base_obs[i : i + 1])
        ref_m = agent.actor_means[t](base_obs[i : i + 1])
    assert torch.allclose(values[i], ref_v[0], atol=1e-6), (i, t, values[i], ref_v)
    assert torch.allclose(means[i], ref_m[0], atol=1e-6), (i, t, means[i], ref_m)
print("nn.1 ok: each row is routed to its own task's network, in the right order")

# 2. Different tasks must give different outputs for the same observation.
same_obs = base_obs[:1].repeat(NUM_TASKS, 1)
oh = torch.eye(NUM_TASKS)
with torch.no_grad():
    v = agent.get_value(torch.cat([oh, same_obs], dim=1)).squeeze(-1)
assert len(set(np.round(v.numpy(), 5))) == NUM_TASKS, v
print("nn.2 ok: identical observations give different values per task")

# 3. The one-hot must not reach the networks: obs_dim excludes it.
assert agent.critics[0][0].in_features == obs_dim, agent.critics[0][0].in_features
assert obs_dim == envs.single_observation_space.shape[0] - NUM_TASKS
print("nn.3 ok: task one-hot is stripped before the networks")

# 4. Independence: a loss on task 0's rows only must not touch tasks 1 and 2.
agent.zero_grad(set_to_none=True)
mask = torch.as_tensor(task_ids == 0)
_, logprob, entropy, value = agent.get_action_and_value(x[mask], torch.zeros(int(mask.sum()), act_dim))
(logprob.sum() + value.sum() + entropy.sum()).backward()
for t in range(NUM_TASKS):
    grads = [p.grad for p in agent.task_parameters(t)]
    touched = any(g is not None and torch.any(g != 0) for g in grads)
    assert touched == (t == 0), f"task {t}: touched={touched}"
print("nn.4 ok: gradients from one task never reach another task's parameters")

# 5. Per-task log-stds are used, and each is its own parameter (so per-task
#    gradient clipping groups are disjoint).
with torch.no_grad():
    for t in range(NUM_TASKS):
        agent.actor_logstds[t].fill_(float(t))
    _, logprob, _, _ = agent.get_action_and_value(x, torch.zeros(B, act_dim))
    mean = agent._dispatch(base_obs, torch.as_tensor(task_ids), agent.actor_means, act_dim)
    for i in range(B):
        t = int(task_ids[i])
        std = torch.exp(torch.tensor(float(t)))
        expected = torch.distributions.Normal(mean[i], std).log_prob(torch.zeros(act_dim)).sum()
        assert torch.allclose(logprob[i], expected, atol=1e-5), (i, t, logprob[i], expected)
ids = {id(p) for p in agent.actor_logstds}
assert len(ids) == NUM_TASKS
print("nn.5 ok: per-task log-stds are used and are disjoint parameters")

# 6. Empty-batch guard.
empty = torch.zeros(0, NUM_TASKS + obs_dim)
assert agent.get_value(empty).shape == (0, 1)
print("nn.6 ok: empty batch handled")

envs.close()


# ==========================================================================
# Multi-task collection loop (ddpg/td3)
# ==========================================================================

import numpy as np
import custom_envs  # noqa: F401

from ddpg_continuous_actions_drats import (
    make_multitask_train_vec_env,
    make_eval_vec_env,
    episode_scalar,
)

NUM_TASKS = 3
NUM_ENVS = 2
envs = make_multitask_train_vec_env(
    env_ids=["Goal2D-v0"] * NUM_TASKS,
    num_envs=NUM_ENVS,
    capture_video=False,
    run_name="t",
    seed=0,
    asynchronous=False,
)
envs.set_task_probs(np.array([0.6, 0.3, 0.1]))
envs.single_action_space.seed(0)

obs, _ = envs.reset(seed=0)
crossings = 0
stored = 0
episodes = 0
task_counts = np.zeros(NUM_TASKS, dtype=int)

for step in range(3000):
    actions = np.array([envs.single_action_space.sample() for _ in range(NUM_ENVS)])
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    if "final_info" in infos:
        for finfo in infos["final_info"]:
            if finfo and "episode" in finfo:
                episodes += 1
                task_counts[int(finfo["task_id"])] += 1
                assert 0.0 <= episode_scalar(finfo.get("is_success", 0.0)) <= 1.0

    # This is the substitution performed in the training scripts.
    real_next_obs = next_obs.copy()
    if "final_observation" in infos:
        for idx, done in enumerate(np.logical_or(terminations, truncations)):
            if done and infos["final_observation"][idx] is not None:
                real_next_obs[idx] = infos["final_observation"][idx]

    for i in range(NUM_ENVS):
        stored += 1
        if obs[i, :NUM_TASKS].argmax() != real_next_obs[i, :NUM_TASKS].argmax():
            crossings += 1

    obs = next_obs

assert episodes > 50, episodes
assert crossings == 0, f"{crossings}/{stored} replay transitions crossed a task boundary"
print(f"env.1 ok: {stored} transitions stored, none crossed a task boundary ({episodes} episodes)")

# The empirical task frequency should follow the sampling distribution.
freq = task_counts / task_counts.sum()
assert np.allclose(freq, [0.6, 0.3, 0.1], atol=0.06), freq
print(f"env.2 ok: episode task frequency {np.round(freq, 3)} follows q = [0.6, 0.3, 0.1]")

# set_task_probs must reach every vector slot.
envs.set_task_probs(np.array([0.1, 0.1, 0.8]))
assert np.allclose(envs.get_task_probs(), [0.1, 0.1, 0.8])
for p in envs.call("get_task_probs"):
    assert np.allclose(p, [0.1, 0.1, 0.8]), p
print(f"env.3 ok: task probs propagate to all {NUM_ENVS} vector slots")

# Eval env: one env per task, fixed one-hot, in order.
envs_eval = make_eval_vec_env(["Goal2D-v0"] * NUM_TASKS, asynchronous=False)
eval_obs, _ = envs_eval.reset(seed=0)
assert eval_obs.shape[0] == NUM_TASKS
for t in range(NUM_TASKS):
    assert eval_obs[t, :NUM_TASKS].argmax() == t, eval_obs[:, :NUM_TASKS]
print("env.4 ok: eval env has exactly one env per task, in task order")

envs.close()
envs_eval.close()

print("\nall DRATS invariant checks passed")
