import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
import torch as th
import argparse
from copy import deepcopy
import os
import sys
from omegaconf import OmegaConf

from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.sac import SAC, MlpPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecCheckNan
from stable_baselines3.common.callbacks import EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback

PROJECT_ROOT_DIR = Path(__file__).parent.parent

if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

import envs

np.seterr(all="raise")  # 检查nan


def train(config, wandb_run):

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / config["algo"]["experiment_name"]).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    # prepare env
    vec_env = make_vec_env(
        env_id=config["env"]["id"],
        env_kwargs=config["env"]["config"] if "config" in config["env"] else None,
        n_envs=config["algo"]["rollout_process_num"],
        seed=config["algo"]["seed_in_training_env"],
        vec_env_cls=SubprocVecEnv,
    )

    print(f"max_episode_steps: {vec_env.get_attr('_max_episode_steps', indices=[0])}, config: {config.env.config if 'config' in config.env else None}")

    eval_env_in_callback = make_vec_env(
        env_id=config["env"]["id"],
        env_kwargs=config["env"]["config"] if "config" in config["env"] else None,
        n_envs=config["algo"]["rollout_process_num"],
        seed=config["algo"]["callback_process_num"],
        vec_env_cls=SubprocVecEnv, 
    )



    # SAC hyperparams:
    sac_algo = SAC(
        MlpPolicy,
        vec_env,
        seed=config["algo"]["seed"],
        replay_buffer_class=HerReplayBuffer if config["algo"]["use_her"] else ReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ) if config["algo"]["use_her"] else None,
        verbose=1,
        buffer_size=int(config["algo"]["buffer_size"]),
        learning_starts=int(config["algo"]["learning_starts"]),
        gradient_steps=int(config["algo"]["gradient_steps"]),
        learning_rate=config["algo"]["learning_rate"],
        gamma=config["algo"]["gamma"],
        batch_size=int(config["algo"]["batch_size"]),
        policy_kwargs=dict(
            # net_arch=config["algo"]["net_arch"],
            net_arch=[128, 128],
            activation_fn=th.nn.Tanh
        ),
    )

    sac_algo.set_logger(sb3_logger)

    # callback: evaluate, save best
    eval_callback = EvalCallback(
        eval_env_in_callback, 
        best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / config["algo"]["experiment_name"]).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "logs" / config["algo"]["experiment_name"]).absolute()), 
        eval_freq=config["algo"]["eval_freq"],  # 多少次env.step()评估一次，此处设置为1000，因为VecEnv有72个并行环境，所以实际相当于72*1000次step，评估一次
        n_eval_episodes=config["algo"]["n_eval_episodes"],  # 每次评估使用多少条轨迹
        deterministic=True, 
        render=False,
    )

    sac_algo.learn(
        total_timesteps=int(config["algo"]["train_steps"]), 
        callback=[
            eval_callback,
            WandbCallback(
                model_save_freq=config["algo"]["save_checkpoint_every_n_timesteps"],
                model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / config["algo"]["experiment_name"] / wandb_run.id).absolute()),
                verbose=2,
            )
        ]
    )
    # sac_algo.save(str(PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME))

    return sb3_logger, vec_env, eval_env_in_callback

if __name__ == "__main__":

    conf = OmegaConf.from_cli()
    
    train_config_dir = conf.config_file

    train_config = OmegaConf.load(train_config_dir)

    print(train_config)

    run = wandb.init(
        project="sim2real",
        config=train_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
        mode="offline",
    )

    sb3_logger, train_env, eval_env = train(train_config, run)

    run.finish()
