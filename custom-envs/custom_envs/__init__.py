import os

from gymnasium.envs.registration import register
from gymnasium_robotics.envs.maze import maps

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

def _merge(a, b):
    a.update(b)
    return a
kwargs = {
    "reward_type": "sparse",
}

register(
    id=f"CleanPointMaze1-v0",
    entry_point="custom_envs.envs.cleanpointmaze:CleanPointMaze",
    kwargs=_merge(
        {
            "maze_map": maps.U_MAZE,
        },
        kwargs,
    ),
    max_episode_steps=300,
)

register(
    id=f"CleanPointMaze2-v0",
    entry_point="custom_envs.envs.cleanpointmaze:CleanPointMaze",
    kwargs=_merge(
        {
            "maze_map": maps.MEDIUM_MAZE,
        },
        kwargs,
    ),
    max_episode_steps=600,
)

register(
    id=f"CleanPointMaze3-v0",
    entry_point="custom_envs.envs.cleanpointmaze:CleanPointMaze",
    kwargs=_merge(
        {
            "maze_map": maps.LARGE_MAZE,
        },
        kwargs,
    ),
    max_episode_steps=800,
)

for i in range(1, 6):
    register(
        id=f"Bandit{i}-v0",
        entry_point=f"custom_envs.envs.bandit:Bandit{i}",
        max_episode_steps=1,
    )

for i in range(1, 5):
    register(
        id=f"Goal2D{i}-v0",
        entry_point=f"custom_envs.envs.goal2d:Goal2D{i}Env",
        max_episode_steps=20,
    )