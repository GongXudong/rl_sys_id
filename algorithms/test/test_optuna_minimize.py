import sys
from pathlib import Path
import unittest
import gymnasium as gym
from omegaconf import OmegaConf
from copy import deepcopy
import numpy as np

from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC

# Add the parent directory to the system path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

import envs
from algorithms.optuna_minimize import SystemIdentificationWithOptuna
from envs.custom_mountain_car_continuous import CustomContinuousMountainCarEnv
from envs.custom_cartpole import CustomCartPoleEnv
from envs.custom_pendulum import CustomPendulumEnv
from utils.collect_samples import collect_samples


class SystemIdentificationWithOptunaTest(unittest.TestCase):

    def setUp(self):
        super().setUp()

        # self.setup_custom_mountain_car()
        # self.setup_custom_pendulum()
        self.setup_custom_cartpole()
    
    def setup_custom_mountain_car(self):
        self.env_config_real = {
            "power": 0.0011
        }
        self.env_id = "CustomMountainCarContinuous-v0"
        self.collect_real_sample_num = 1000
        self.helper_env_class = CustomContinuousMountainCarEnv
        self.policy_class = PPO
        self.policy_path = PROJECT_ROOT_DIR / "checkpoints/custom_continuous_mountain_car/ppo/power_0_0015/best_model.zip"
        self.current_params = {
            "power": 0.0015,  # 当前参数
        }
        self.params_config = {
            "power": {
                "range": [0.001, 0.002],
            }
        }
        self.optimize_n_iter = 100
        self.seed_optimize = 33
    
    def setup_custom_pendulum(self):
        self.env_config_real = {
            "g": 8.0,
            "m": 1.0,
            "l": 1.0,
        }
        self.env_id = "CustomPendulum-v0"
        self.collect_real_sample_num = 10000
        self.helper_env_class = CustomPendulumEnv
        self.policy_class = PPO
        self.policy_path = PROJECT_ROOT_DIR / "checkpoints/custom_pendulum/g_10_0_m_1_0_l_1_0/ppo/seed_1/best_model.zip"
        self.current_params = {
            "g": 10.0,
            "m": 1.0,
            "l": 1.0,
        }
        self.params_config = {
            # "g": {
            #     "range": [8.0, 12.0],
            # },
            "m": {
                "range": [0.5, 1.5],
            },
            # "l": {
            #     "range": [0.5, 1.5],
            # }
        }
        self.optimize_n_iter = 100
        self.seed_optimize = 333234
    
    def setup_custom_cartpole(self):
        self.env_config_real = {
            "gravity": 9.8,
            "masscart": 1.5,
            "masspole": 0.05,
            "length": 0.5,
            "force_mag": 10.0,
            "tau": 0.02,
        }
        self.env_id = "CustomCartPole-v0"
        self.collect_real_sample_num = 3000
        self.helper_env_class = CustomCartPoleEnv
        self.policy_class = PPO
        self.policy_path = PROJECT_ROOT_DIR / "checkpoints/custom_cartpole/ppo/best_model.zip"
        self.current_params = {
            "gravity": 9.8,
            "masscart": 1.5,
            "masspole": 0.1,
            "length": 0.5,
            "force_mag": 10.0,
            "tau": 0.02,
        }
        self.params_config = {
            "masscart": {
                "range": [0.5, 1.5],
            },
            "masspole": {
                "range": [0.05, 0.15],
            },
        }
        self.optimize_n_iter = 1000
        self.seed_optimize = 86003

    def test_init_1(self):
        print(f"test init 2")
        algo = SystemIdentificationWithOptuna(
            current_params=self.current_params,
            params_config=self.params_config,
            helper_env_class=self.helper_env_class,
        )

        env_real = gym.make(self.env_id, **self.env_config_real)
        helper_env = algo.helper_env_class.get_env_from_config(config=env_real.unwrapped.get_config())

        self.assertDictEqual(env_real.unwrapped.get_config(), helper_env.get_config(), "Environment configuration does not match helper environment configuration.")

        obs, info = env_real.reset()

        for i in range(10):
            action = env_real.action_space.sample()
            
            next_obs, reward, terminated, truncated, info = env_real.step(action)
            # print(f"next_obs from env: {next_obs}")
            
            next_obs_calculated_from_dynamics = algo.helper_env_class.calc_next_obs(state=obs, action=action, helper_env=helper_env)
            # print(f"next_obs from dynamics: {next_obs_calculated_from_dynamics}")

            assert np.allclose(next_obs, next_obs_calculated_from_dynamics), f"Observation mismatch at step {i+1}: {next_obs} != {next_obs_calculated_from_dynamics}"

            obs = next_obs

            if terminated or truncated:
                obs, info = env_real.reset()
    
    def test_optimize(self):
        print(f"test optimize")

        env_real = self.helper_env_class.get_env_from_config(config=self.env_config_real)
        
        # load policy trained on sim environment
        algo = self.policy_class.load(self.policy_path, env=env_real)

        obs_real, act_real, next_obs_real = collect_samples(
            policy=algo.policy,
            env=env_real,
            num_samples=self.collect_real_sample_num,
            deterministic=True,
        )

        algo = SystemIdentificationWithOptuna(
            current_params=self.current_params,
            params_config=self.params_config,
            helper_env_class=self.helper_env_class,
        )

        # 执行优化
        study = algo.optimize(obs_real=obs_real, act_real=act_real, next_obs_real=next_obs_real, n_iter=self.optimize_n_iter, seed=self.seed_optimize)

        print(f"Best parameters found: {study.best_params}")

if __name__ == "__main__":
    unittest.main()
