from gymnasium.envs.registration import register
from sensor_interaction.autorl_world import Auto_RL
from sensor_interaction.MainNode import MainNode


def create_auto_rl_env(**kwargs):
    main_node = MainNode()
    return Auto_RL(main_node=main_node)


# Register the custom environment
register(
    id="Autopilot-RL-v0",
    entry_point="sensor_interaction:create_auto_rl_env",
    max_episode_steps=400,
)
