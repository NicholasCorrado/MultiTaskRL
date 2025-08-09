import os

from gymnasium.envs.registration import register

ENVS_DIR = os.path.join(os.path.dirname(__file__), 'envs')

############################################################################
### Toy

register(
    id="Goal2D-v0",
    entry_point="custom_envs.envs.goal2d:Goal2DEnv",
    max_episode_steps=100,
)

register(
    id="BanditEasy-v0",
    entry_point="custom_envs.envs.bandit:BanditEasy",
    max_episode_steps=1,
)

register(
    id="BanditHard-v0",
    entry_point="custom_envs.envs.bandit:BanditHard",
    max_episode_steps=1,
)

register(
    id="GridWorldEnv-v0",
    entry_point="custom_envs.envs.gridworld:GridWorldEnv",
)

for i in range(1, 6):
    register(
        id=f"Bandit{i}-v0",
        entry_point=f"custom_envs.envs.bandit:Bandit{i}",
        max_episode_steps=1,
    )


for i in range(1, 5):
    register(
        id=f"GridWorldEnv{i}-v0",
        entry_point=f"custom_envs.envs.gridworld:GridWorldEnv{i}",
        max_episode_steps=10,
    )

for i in range(1, 5):
    register(
        id=f"Goal2D{i}-v0",
        entry_point=f"custom_envs.envs.goal2d:Goal2D{i}Env",
        max_episode_steps=100,
    )

register(
    id=f"Goal2DEasy-v0",
    entry_point=f"custom_envs.envs.goal2d:Goal2DEasyEnv",
    max_episode_steps=40,
)

register(
    id=f"Goal2DHard-v0",
    entry_point=f"custom_envs.envs.goal2d:Goal2DHardEnv",
    max_episode_steps=40,
)

for i in range(1, 4):
    register(
        id=f"PointMaze{i}-v0",
        entry_point=f"custom_envs.envs.pointmaze:PointMazeEnv{i}",
        max_episode_steps=100,
    )