import os

from gymnasium.envs.registration import register

from custom_envs.envs.gridworld import gridworld_maps

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

# Register environments
for i, gmap in enumerate(gridworld_maps):
    register(
        id=f"GridWorld{i}-v0",
        entry_point="custom_envs.envs.gridworld:GridWorldEnv",
        max_episode_steps=15,
        kwargs={
            "map": gmap
        }
    )