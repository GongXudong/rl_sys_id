import sys
from pathlib import Path
import unittest
import gymnasium as gym
import numpy as np

# Add the parent directory to the system path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from envs.sys_id_env import SystemIdentificationEnv
from utils.wrappers.multibinary_to_discrete import MultiBinaryToDiscreteWrapper


class MultiBinaryToDiscreteWrapperTest(unittest.TestCase):

    def setUp(self):
        
        self.params_config = {
            "g": {
                "initial_value": 9.8,
                "optimize": True,
                "range": [9.5, 10.0],
            },
            "m": {
                "initial_value": 1.0,
                "optimize": True,
                "range": [0.5, 1.5],
            },
            "l": {
                "initial_value": 1.1,
                "optimize": True,
                "range": [0.5, 1.5],
            },
        }

        self.env = SystemIdentificationEnv(
            dynamics_env_id="CustomPendulum-v0",
            params_config=self.params_config,
            obs_real_file_path=PROJECT_ROOT_DIR / "data/custom_pendulum/obs_real.npy",
            act_real_file_path=PROJECT_ROOT_DIR / "data/custom_pendulum/act_real.npy",
            next_obs_real_file_path=PROJECT_ROOT_DIR / "data/custom_pendulum/next_obs_real.npy",
            bo_optimizer_n_trials=30,
            bo_optimizer_n_jobs=-1,
            reward_b=1.0,
            max_steps=30,
            loss_threshold=1e-10,
        )
        self.wrapped_env = MultiBinaryToDiscreteWrapper(self.env)

    def test_1(self):
        print(self.wrapped_env._action_mapping)
        print(self.wrapped_env.action_space)

        # for i in range(30):
        #     print(self.wrapped_env.action_space.sample())


if __name__ == "__main__":
    unittest.main()
    