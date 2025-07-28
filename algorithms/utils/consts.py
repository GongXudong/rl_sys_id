from envs.custom_cartpole import CustomCartPoleEnv
from envs.custom_mountain_car_continuous import CustomContinuousMountainCarEnv
from envs.custom_pendulum import CustomPendulumEnv


HELPER_ENV_CLASS_DICT = {
    "CustomCartPole-v0": CustomCartPoleEnv,
    "CustomMountainCarContinuous-v0": CustomContinuousMountainCarEnv,
    "CustomPendulum-v0": CustomPendulumEnv,
}