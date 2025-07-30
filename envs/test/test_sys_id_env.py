import sys
from pathlib import Path
import unittest
import gymnasium as gym
from omegaconf import OmegaConf
from copy import deepcopy
import numpy as np
import optuna
from time import time

# Add the parent directory to the system path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

import envs
from envs.sys_id_env import SystemIdentificationEnv


optuna.logging.set_verbosity(optuna.logging.WARNING)  # 关闭Optuna控制台的输出


class SystemIdentificationEnvTest(unittest.TestCase):

    def setUp(self):
        self.params_config = {
            "gravity": {
                "initial_value": 9.8,
                "optimize": True,
                "range": [9.5, 10.0],
            },
            "masscart": {
                "initial_value": 1.0,
                "optimize": True,
                "range": [0.5, 1.5],
            },
            "masspole": {
                "initial_value": 0.1,
                "optimize": True,
                "range": [0.05, 0.15],
            },
            "length": {
                "initial_value": 0.5,
                "optimize": True,
                "range": [0.5, 1.0],
            },
            "force_mag": {
                "initial_value": 10.0,
                "optimize": True,
                "range": [9.0, 11.0],
            },
            "tau": {
                "initial_value": 0.02,
                "optimize": False,  # 不需要优化
                "range": [0.01, 0.03],
            },
        }

        self.env = SystemIdentificationEnv(
            dynamics_env_id="CustomCartPole-v0",
            params_config=self.params_config,
            obs_real_file_path=PROJECT_ROOT_DIR / "data/custom_cartpole/obs_real.npy",
            act_real_file_path=PROJECT_ROOT_DIR / "data/custom_cartpole/act_real.npy",
            next_obs_real_file_path=PROJECT_ROOT_DIR / "data/custom_cartpole/next_obs_real.npy",
            bo_optimizer_n_trials=30,
            bo_optimizer_seed=42,
            bo_optimizer_threads=-1,
            reward_b=1.0,
            max_steps=30,
            loss_threshold=1e-10,
        )

    def test_init_1(self):
        # test observation_space and action_space initialization
        print("test init 1:")

        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")

        self.assertTrue(np.allclose(self.env.observation_space.low, [self.params_config[ky]["range"][0] for ky in self.params_config.keys() if self.params_config[ky].get("optimize", False)]))
        self.assertTrue(np.allclose(self.env.observation_space.high, [self.params_config[ky]["range"][1] for ky in self.params_config.keys() if self.params_config[ky].get("optimize", False)]))

        self.assertListEqual(list(self.env.key_list), list(self.params_config.keys()), "key_list does not match expected values.")
        self.assertListEqual(list(self.env.key_list_of_params_to_be_optimized), [ky for ky in self.params_config.keys() if self.params_config[ky]["optimize"]], "key_list_of_params_to_be_optimized does not match expected values.")
    
    def test_get_params_to_be_optimized_from_action_1(self):
        # test get_params_to_be_optimized_from_action method
        print("test get_params_to_be_optimized_from_action 1:")
        action = np.array([1, 0, 1, 0, 1])
        self.env.reset()
        expected_params = self.env.get_params_to_be_optimized_from_action(self.env.current_params_to_be_optimized, action)
        print(expected_params)
        self.assertListEqual(list(expected_params.keys()), ["gravity", "masspole", "force_mag"], "Expected parameters to be optimized do not match the action taken.")

    def test_reset_1(self):
        # test reset method
        print("test reset 1:")
        obs, info = self.env.reset()
        self.assertTrue(np.allclose(obs, list([self.params_config[ky]["initial_value"] for ky in self.params_config.keys() if self.params_config[ky].get("optimize", False)]), atol=1e-6), "Reset observation does not match current configuration.")
        print(f"Initial loss: {info["loss"]}")

    def test_step_1(self):
        # test step method
        print("test step 1:")
        obs, info = self.env.reset()
        for i in range(35):
            step_start_time = time()
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            step_end_time = time()
            print(f"Step {i+1}:")
            print(f"Action: {action}")
            print(f"Next observation: {next_obs}")
            print(f"Reward: {reward}")
            print(f"Info: {info}")
            print(f"Time: {(step_end_time - step_start_time)}")

            if terminated or truncated:
                obs, info = self.env.reset()
                print("Environment reset.")

if __name__ == "__main__":
    unittest.main()
