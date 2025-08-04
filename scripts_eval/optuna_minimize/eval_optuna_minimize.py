import sys
from pathlib import Path
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
from envs.sys_id_env import SystemIdentificationEnv
from utils.wrappers.multibinary_to_discrete import MultiBinaryToDiscreteWrapper


def work(config):

    training_env = SystemIdentificationEnv(
        dynamics_env_id=config["dynamics_env"]["id"],
        params_config=config["dynamics_env"]["params_config"],
        obs_real_file_path=PROJECT_ROOT_DIR / config["data_collected_from_real"]["obs_file_path"],
        act_real_file_path=PROJECT_ROOT_DIR / config["data_collected_from_real"]["act_file_path"],
        next_obs_real_file_path=PROJECT_ROOT_DIR / config["data_collected_from_real"]["next_obs_file_path"],
        bo_optimizer_n_trials=config["bo_optimizer"]["n_trials"],
        bo_optimizer_n_jobs=config["bo_optimizer"]["n_jobs"],
        bo_optimizer_sample_num_in_optimize=config["bo_optimizer"]["sample_num_in_optimize"],
        reward_b=config["sys_id_env"]["reward_b"],
        max_steps=config["sys_id_env"]["max_steps"],
        loss_threshold=config["sys_id_env"]["loss_threshold"],
    )
    training_env = MultiBinaryToDiscreteWrapper(training_env)


    sys_id_algo = SystemIdentificationWithOptuna(
        current_params=self.current_params,
        params_config=self.params_config,
        helper_env_class=self.helper_env_class,
    )

    # 执行优化
    study = sys_id_algo.optimize(
        obs_real=obs_real, 
        act_real=act_real, 
        next_obs_real=next_obs_real,
        n_trials=self.optimize_n_trials,
        n_jobs=self.n_jobs,
        seed=self.seed_optimize
    )
    print()


if __name__ == "__main__":

    conf = OmegaConf.from_cli()
    
    train_config_dir = conf.config_file

    train_config = OmegaConf.load(train_config_dir)
    
    work(train_config)
