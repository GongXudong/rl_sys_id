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
from envs.custom_cartpole import CustomCartPoleEnv


class CustomCartPoleTest(unittest.TestCase):

    def test_init_1(self):
        print("test init 1:")

        gravity = 9.5
        masscart = 1.1
        masspole = 0.2
        length = 0.6
        force_mag = 10.5
        tau = 0.03

        env = gym.make("CustomCartPole-v0", gravity=gravity, masscart=masscart, masspole=masspole, length=length, force_mag=force_mag, tau=tau)

        self.assertEqual(env.unwrapped.gravity, gravity, f"graviry not equal: {env.unwrapped.gravity}, {gravity}")
        self.assertEqual(env.unwrapped.masscart, masscart, f"masscart not equal: {env.unwrapped.masscart}, {masscart}")
        self.assertEqual(env.unwrapped.masspole, masspole, f"masspole not equal: {env.unwrapped.masspole}, {masspole}")
        self.assertEqual(env.unwrapped.length, length, f"length not equal: {env.unwrapped.length}, {length}")
        self.assertEqual(env.unwrapped.force_mag, force_mag, f"force_mag not equal: {env.unwrapped.force_mag}, {force_mag}")
        self.assertEqual(env.unwrapped.tau, tau, f"tau not equal: {env.unwrapped.tau}, {tau}")

    def test_init_2(self):
        print("test init 2:")

        conf_dir = PROJECT_ROOT_DIR / "envs" / "test" / "custom_cartpole_config.yaml"
        conf = OmegaConf.load(conf_dir)

        env = gym.make(id=conf["env"]["id"], **conf["env"]["config"])

        self.assertEqual(env.unwrapped.gravity, conf.env.config.gravity, f"graviry not equal: {env.unwrapped.gravity}, {conf.env.config.gravity}")
        self.assertEqual(env.unwrapped.masscart, conf.env.config.masscart, f"masscart not equal: {env.unwrapped.masscart}, {conf.env.config.masscart}")
        self.assertEqual(env.unwrapped.masspole, conf.env.config.masspole, f"masspole not equal: {env.unwrapped.masspole}, {conf.env.config.masspole}")
        self.assertEqual(env.unwrapped.length, conf.env.config.length, f"length not equal: {env.unwrapped.length}, {conf.env.config.length}")
        self.assertEqual(env.unwrapped.force_mag, conf.env.config.force_mag, f"force_mag not equal: {env.unwrapped.force_mag}, {conf.env.config.force_mag}")
        self.assertEqual(env.unwrapped.tau, conf.env.config.tau, f"tau not equal: {env.unwrapped.tau}, {conf.env.config.tau}")

    def test_reset_1(self):
        print("test reset:")

        gravity_1 = 9.5
        masscart_1 = 1.1
        masspole_1 = 0.2
        length_1 = 0.6
        force_mag_1 = 10.5
        tau_1 = 0.03

        gravity_2 = 9.5
        masscart_2 = 1.1
        masspole_2 = 0.2
        length_2 = 0.6
        force_mag_2 = 10.5
        tau_2 = 0.03

        env_1 = gym.make("CustomCartPole-v0", gravity=gravity_1, masscart=masscart_1, masspole=masspole_1, length=length_1, force_mag=force_mag_1, tau=tau_1)
        env_2 = gym.make("CustomCartPole-v0", gravity=gravity_2, masscart=masscart_2, masspole=masspole_2, length=length_2, force_mag=force_mag_2, tau=tau_2)

        obs_1, info_1 = env_1.reset()
        obs_2, info_2 = env_2.reset()

        print(obs_1)
        print(obs_2)

    def test_step_1(self):
        print("test step 1:")

        gravity_1 = 9.5
        masscart_1 = 1.1
        masspole_1 = 0.2
        length_1 = 0.6
        force_mag_1 = 10.5
        tau_1 = 0.03

        env_1 = gym.make("CustomCartPole-v0", gravity=gravity_1, masscart=masscart_1, masspole=masspole_1, length=length_1, force_mag=force_mag_1, tau=tau_1)

        obs_1, info_1 = env_1.reset()
        steps = 0

        while True:
            action = env_1.action_space.sample()
            obs_1, reward_1, terminated_1, truncated_1, info_1 = env_1.step(action)
            steps += 1
            print(obs_1, reward_1, terminated_1, truncated_1, info_1)
            if terminated_1 or truncated_1:
                break
        
        print(f"steps: {steps}")

    def test_merge_config_with_default_config_1(self):
        print("test merge_config_with_default_config 1:")

        custom_config = {
            "gravity": 9.5,
            "masscart": 1.1,
            "masspole": 0.2,
            "length": 0.6,
            "force_mag": 10.5,
            "tau": 0.03
        }

        merged_config = CustomCartPoleEnv.merge_config_with_default_config(custom_config)

        self.assertDictEqual(merged_config, custom_config, "Merged configuration does not match the custom configuration.")

    def test_merge_config_with_default_config_2(self):
        print("test merge_config_with_default_config 2:")

        custom_config = {
            "gravity": 8.0,
        }

        merged_config = CustomCartPoleEnv.merge_config_with_default_config(custom_config)

        default_config = CustomCartPoleEnv.get_default_config()
        for key, value in custom_config.items():
            default_config[key] = value

        self.assertDictEqual(merged_config, default_config, "Merged configuration does not match the custom configuration.")

    def test_get_env_from_config_1(self):
        print("test get_env_from_config 1:")

        custom_config = {
            "gravity": 9.5,
            "masscart": 1.1,
            "masspole": 0.2,
            "length": 0.6,
            "force_mag": 10.5,
            "tau": 0.03
        }

        env = CustomCartPoleEnv.get_env_from_config(config=custom_config)

        self.assertEqual(env.unwrapped.gravity, custom_config["gravity"], f"graviry not equal: {env.unwrapped.gravity}, {custom_config["gravity"]}")
        self.assertEqual(env.unwrapped.masscart, custom_config["masscart"], f"masscart not equal: {env.unwrapped.masscart}, {custom_config["masscart"]}")
        self.assertEqual(env.unwrapped.masspole, custom_config["masspole"], f"masspole not equal: {env.unwrapped.masspole}, {custom_config["masspole"]}")
        self.assertEqual(env.unwrapped.length, custom_config["length"], f"length not equal: {env.unwrapped.length}, {custom_config["length"]}")
        self.assertEqual(env.unwrapped.force_mag, custom_config["force_mag"], f"force_mag not equal: {env.unwrapped.force_mag}, {custom_config["force_mag"]}")
        self.assertEqual(env.unwrapped.tau, custom_config["tau"], f"tau not equal: {env.unwrapped.tau}, {custom_config["tau"]}")

    def test_get_env_from_config_2(self):
        print("test get_env_from_config 2:")

        custom_config = {
            "gravity": 9.5,
            "masscart": 1.1,
            "masspole": 0.2,
            "length": 0.6,
            "force_mag": 10.5,
            "tau": 0.03
        }

        env1 = CustomCartPoleEnv.get_env_from_config(config=custom_config)

        self.assertEqual(env1.unwrapped.gravity, custom_config["gravity"], f"graviry not equal: {env1.unwrapped.gravity}, {custom_config["gravity"]}")
        self.assertEqual(env1.unwrapped.masscart, custom_config["masscart"], f"masscart not equal: {env1.unwrapped.masscart}, {custom_config["masscart"]}")
        self.assertEqual(env1.unwrapped.masspole, custom_config["masspole"], f"masspole not equal: {env1.unwrapped.masspole}, {custom_config["masspole"]}")
        self.assertEqual(env1.unwrapped.length, custom_config["length"], f"length not equal: {env1.unwrapped.length}, {custom_config["length"]}")
        self.assertEqual(env1.unwrapped.force_mag, custom_config["force_mag"], f"force_mag not equal: {env1.unwrapped.force_mag}, {custom_config["force_mag"]}")
        self.assertEqual(env1.unwrapped.tau, custom_config["tau"], f"tau not equal: {env1.unwrapped.tau}, {custom_config["tau"]}")

        custom_config["gravity"] = 9.8
        custom_config["masscart"] = 1.2
        custom_config["masspole"] = 0.3
        custom_config["length"] = 0.7
        custom_config["force_mag"] = 11.0
        custom_config["tau"] = 0.04

        env2 = CustomCartPoleEnv.get_env_from_config(config=custom_config)

        self.assertEqual(env2.unwrapped.gravity, custom_config["gravity"], f"graviry not equal: {env2.unwrapped.gravity}, {custom_config["gravity"]}")
        self.assertEqual(env2.unwrapped.masscart, custom_config["masscart"], f"masscart not equal: {env2.unwrapped.masscart}, {custom_config["masscart"]}")
        self.assertEqual(env2.unwrapped.masspole, custom_config["masspole"], f"masspole not equal: {env2.unwrapped.masspole}, {custom_config["masspole"]}")
        self.assertEqual(env2.unwrapped.length, custom_config["length"], f"length not equal: {env2.unwrapped.length}, {custom_config["length"]}")
        self.assertEqual(env2.unwrapped.force_mag, custom_config["force_mag"], f"force_mag not equal: {env2.unwrapped.force_mag}, {custom_config["force_mag"]}")
        self.assertEqual(env2.unwrapped.tau, custom_config["tau"], f"tau not equal: {env2.unwrapped.tau}, {custom_config["tau"]}")

        self.assertNotEqual(env1.unwrapped.gravity, env2.unwrapped.gravity, f"gravity should not be equal: {env1.unwrapped.gravity}, {env2.unwrapped.gravity}")
        self.assertNotEqual(env1.unwrapped.masscart, env2.unwrapped.masscart, f"masscart should not be equal: {env1.unwrapped.masscart}, {env2.unwrapped.masscart}")
        self.assertNotEqual(env1.unwrapped.masspole, env2.unwrapped.masspole, f"masspole should not be equal: {env1.unwrapped.masspole}, {env2.unwrapped.masspole}")
        self.assertNotEqual(env1.unwrapped.length, env2.unwrapped.length, f"length should not be equal: {env1.unwrapped.length}, {env2.unwrapped.length}")
        self.assertNotEqual(env1.unwrapped.force_mag, env2.unwrapped.force_mag, f"force_mag should not be equal: {env1.unwrapped.force_mag}, {env2.unwrapped.force_mag}")
        self.assertNotEqual(env1.unwrapped.tau, env2.unwrapped.tau, f"tau should not be equal: {env1.unwrapped.tau}, {env2.unwrapped.tau}")

    def test_calc_next_obs_1(self):
        print("test calc_next_obs 1:")

        gravity_1 = 9.5
        masscart_1 = 1.1
        masspole_1 = 0.2
        length_1 = 0.6
        force_mag_1 = 10.5
        tau_1 = 0.03

        env_1 = gym.make("CustomCartPole-v0", gravity=gravity_1, masscart=masscart_1, masspole=masspole_1, length=length_1, force_mag=force_mag_1, tau=tau_1)
        env_2 = gym.make("CustomCartPole-v0", gravity=gravity_1, masscart=masscart_1, masspole=masspole_1, length=length_1, force_mag=force_mag_1, tau=tau_1)

        for i in range(10):
            obs_1, info_1 = env_1.reset()
            obs_2, info_2 = env_2.reset()

            random_steps = np.random.randint(5, 20)
            for i in range(random_steps):
                prev_obs_1 = deepcopy(obs_1)
                action_1 = env_1.action_space.sample()
                obs_1, reward_1, terminated_1, truncated_1, info_1 = env_1.step(action_1)
                if terminated_1 or truncated_1:
                    break

            obs_2 = CustomCartPoleEnv.calc_next_obs(prev_obs_1, action_1, env_2)

            self.assertAlmostEqual(obs_1[0], obs_2[0], delta=1e-5, msg=f"obs_1[0] not equal: {obs_1[0]}, {obs_2[0]}")
            self.assertAlmostEqual(obs_1[1], obs_2[1], delta=1e-5, msg=f"obs_1[1] not equal: {obs_1[1]}, {obs_2[1]}")
            self.assertAlmostEqual(obs_1[2], obs_2[2], delta=1e-5, msg=f"obs_1[2] not equal: {obs_1[2]}, {obs_2[2]}")
            self.assertAlmostEqual(obs_1[3], obs_2[3], delta=1e-5, msg=f"obs_1[3] not equal: {obs_1[3]}, {obs_2[3]}")


if __name__ == "__main__":
    unittest.main()
