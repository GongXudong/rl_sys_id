import gymnasium as gym
from gymnasium.core import ObsType, ActType
from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from copy import deepcopy

from envs.utils.env_config_mixins import EnvConfigMixin, DynamicsMixin


class CustomContinuousMountainCarEnv(Continuous_MountainCarEnv, EnvConfigMixin, DynamicsMixin):

    def __init__(self, render_mode = None, goal_velocity=0, power: float=0.0015):
        super().__init__(render_mode, goal_velocity)
        
        self.set_config({
            "power": power
        })

    def get_config(self) -> dict:
        return {
            "power": self.power
        }

    def set_config(self, config: dict):
        self.power = config.get("power", 0.0015)

    @staticmethod
    def get_env_id() -> str:
        return "CustomMountainCarContinuous-v0"

    @classmethod
    def get_env_from_config(cls, *args, config, **kwargs):
        # return cls(*args, **config, **kwargs)
        return gym.make(cls.get_env_id(), **config, **kwargs)

    @staticmethod
    def get_default_config() -> dict:
        return {
            "power": 0.0015
        }

    @staticmethod
    def calc_next_obs(state: ObsType, action: ActType, helper_env: gym.Env) -> ObsType:
        assert isinstance(helper_env.unwrapped, Continuous_MountainCarEnv), "helper_env must be an instance of Continuous_MountainCarEnv"

        helper_env.reset()
        helper_env.unwrapped.state = deepcopy(state)
        next_obs, reward, terminated, truncated, info = helper_env.step(action)
        return next_obs