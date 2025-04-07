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
