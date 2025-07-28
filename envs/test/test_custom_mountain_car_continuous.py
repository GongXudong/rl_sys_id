import sys
from pathlib import Path
import unittest
import gymnasium as gym
from omegaconf import OmegaConf
from copy import deepcopy
import numpy as np

# Add the parent directory to the system path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

import envs
from envs.custom_mountain_car_continuous import CustomContinuousMountainCarEnv


class CustomContinuousMountainCarTest(unittest.TestCase):

    def test_init_1(self):
        print("test init 1:")

        power = 0.002

        env = gym.make("CustomMountainCarContinuous-v0", power=power)

        self.assertEqual(env.unwrapped.power, power, f"power not equal: {env.unwrapped.power}, {power}")
    
    def test_init_2(self):
        print("test init 2:")

        conf_dir = PROJECT_ROOT_DIR / "envs" / "test" / "custom_mountain_car_continuous_config.yaml"
        conf = OmegaConf.load(conf_dir)

        env = gym.make(id=conf["env"]["id"], **conf["env"]["config"])

        self.assertEqual(env.unwrapped.power, conf.env.config.power, f"g not equal: {env.unwrapped.power}, {conf.env.config.power}")
    
    def test_reset_1(self):
        print("test reset:")

        power = 0.002

        env_1 = gym.make("CustomMountainCarContinuous-v0", power=power)
        obs_1, info_1 = env_1.reset()

    def test_get_env_from_config_1(self):
        print("test get_env_from_config 1:")

        custom_config = {
            "power": 0.002
        }

        env = CustomContinuousMountainCarEnv.get_env_from_config(config=custom_config)

        self.assertEqual(env.unwrapped.power, custom_config["power"], f"power not equal: {env.unwrapped.power}, {custom_config["power"]}")

    def test_get_env_from_config_2(self):
        print("test get_env_from_config 2:")

        custom_config = {
            "power": 0.002
        }

        env1 = CustomContinuousMountainCarEnv.get_env_from_config(config=custom_config)

        self.assertEqual(env1.unwrapped.power, custom_config["power"], f"power not equal: {env1.unwrapped.power}, {custom_config["power"]}")

        custom_config["power"] = 0.015

        env2 = CustomContinuousMountainCarEnv.get_env_from_config(config=custom_config)

        self.assertEqual(env2.unwrapped.power, custom_config["power"], f"power not equal: {env2.unwrapped.power}, {custom_config["power"]}")

        self.assertNotEqual(env1.unwrapped.power, env2.unwrapped.power, f"env1 and env2 should have different power values: {env1.unwrapped.power}, {env2.unwrapped.power}")

    def test_calc_next_obs_1(self):
        print("test calc_next_obs 1:")

        conf_dir = PROJECT_ROOT_DIR / "envs" / "test" / "custom_mountain_car_continuous_config.yaml"
        conf = OmegaConf.load(conf_dir)

        env = gym.make(id=conf["env"]["id"], **conf["env"]["config"])
        helper_env = gym.make(id=conf["env"]["id"], **conf["env"]["config"])    

        obs, info = env.reset()

        for _ in range(10):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)

            next_obs_2 = CustomContinuousMountainCarEnv.calc_next_obs(obs, action, helper_env)

            self.assertTrue(np.allclose(next_obs, next_obs_2), f"next_obs not equal: {next_obs}, {next_obs_2}")

            obs = next_obs


if __name__ == "__main__":
    unittest.main()