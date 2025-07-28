import sys
from pathlib import Path
import unittest
import gymnasium as gym
from omegaconf import OmegaConf
from copy import deepcopy
import numpy as np

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.ppo import PPO

# Add the parent directory to the system path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

import envs
from envs.custom_cartpole import CustomCartPoleEnv
from utils.collect_samples import collect_samples


class CollectSamplesTest(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.env_config = {
            "gravity": 9.8,
            "masscart": 1.0,
            "masspole": 0.1,
            "length": 0.5,
            "force_mag": 10.0,
            "tau": 0.02,
        }
        self.env_id = "CustomCartPole-v0"
        self.env = gym.make(self.env_id, **self.env_config)

    def test_1(self):
        print(f"test init 1")

        algo = PPO.load(PROJECT_ROOT_DIR / "checkpoints/custom_cartpole/ppo/best_model.zip")

        obs, acts, next_obs = collect_samples(
            policy=algo.policy,
            env=self.env,
            num_samples=100,
            deterministic=True,
        )

        # 只有在环境没有reset时成立
        for i in range(len(obs)-1):
            self.assertTrue(
                np.allclose(next_obs[i], obs[i+1], atol=1e-6, rtol=1e-6),
                msg=f"Observation mismatch at index {i}: {next_obs[i]} vs {obs[i+1]}"
            )
            print(obs[i], acts[i], next_obs[i])

if __name__ == "__main__":
    unittest.main()
