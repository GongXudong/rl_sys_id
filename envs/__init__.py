from gymnasium.envs.registration import register


register(
    id="CustomCartPole-v0",
    entry_point="envs.custom_cartpole:CustomCartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="CustomMountainCarContinuous-v0",
    entry_point="envs.custom_mountain_car_continuous:CustomContinuousMountainCarEnv",
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id="CustomPendulum-v0",
    entry_point="envs.custom_pendulum:CustomPendulumEnv",
    max_episode_steps=200,
)