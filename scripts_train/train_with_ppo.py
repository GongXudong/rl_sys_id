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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import PPO, MlpPolicy
from stable_baselines3.common.vec_env import VecCheckNan
from stable_baselines3.common.callbacks import EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback

PROJECT_ROOT_DIR = Path(__file__).parent.parent

if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

import envs

np.seterr(all="raise")  # 检查nan

def get_ppo_algo(env, config):
    policy_kwargs = dict(
        net_arch=dict(
            pi=config["algo"]["net_arch"],
            vf=deepcopy(config["algo"]["net_arch"])
        )
    )

    return PPO(
        policy=MlpPolicy, 
        env=env, 
        seed=config["algo"]["seed"],
        batch_size=int(config["algo"]["batch_size"]),
        gamma=config["algo"]["gamma"],
        n_steps=config["algo"]["n_steps"],  # 采样时每个环境采样的step数
        n_epochs=config["algo"]["n_epochs"],  # 采样的数据在训练中重复使用的次数
        ent_coef=config["algo"]["ent_coef"],
        policy_kwargs=policy_kwargs,
        use_sde=config["algo"]["use_sde"],  # 使用state dependant exploration,
        normalize_advantage=True,
        learning_rate=config["algo"]["learning_rate"],
        device=config["algo"]["device"],
    )

def train(config: OmegaConf, wandb_run):
    #register_my_env(goal_range=config["env"]["goal_range"], distance_threshold=config["env"]["distance_threshold"], max_episode_steps=config["env"]["max_episode_steps"])  # 注意此处：max_episode_steps, 根据环境文件的配置修改此值！！！！
    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / config["algo"]["experiment_name"]).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    # prepare env
    vec_env = make_vec_env(
        env_id=config["env"]["id"],
        env_kwargs=config["env"]["config"] if "config" in config["env"] else None,
        n_envs=config["algo"]["rollout_process_num"],
        seed=config["algo"]["seed_in_training_env"],
        vec_env_cls=SubprocVecEnv,
    )

    if config["algo"]["normalize_reward"]:
        vec_env = VecNormalize(
            vec_env, 
            norm_obs=False, 
            norm_reward=True, 
            clip_obs=10.0,
            gamma=config["algo"]["gamma"],
        )

    print(f"max_episode_steps: {vec_env.get_attr('_max_episode_steps', indices=[0])}, config: {config.env.config if 'config' in config.env else None}")

    eval_env_in_callback = make_vec_env(
        env_id=config["env"]["id"],
        env_kwargs=config["env"]["config"] if "config" in config["env"] else None,
        n_envs=config["algo"]["rollout_process_num"],
        seed=config["algo"]["callback_process_num"],
        vec_env_cls=SubprocVecEnv, 
    )

    algo_ppo = get_ppo_algo(vec_env, config)
    sb3_logger.info(str(algo_ppo.policy))

    # set sb3 logger
    algo_ppo.set_logger(sb3_logger)

    eval_callback = EvalCallback(
        eval_env_in_callback, 
        best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / config["algo"]["experiment_name"]).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "logs"/ config["algo"]["experiment_name"]).absolute()), 
        eval_freq=config["algo"]["evaluate_frequence"],
        n_eval_episodes=config["algo"]["evaluate_nums_in_callback"],
        deterministic=True, 
        render=False,
    )

    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path=str((PROJECT_ROOT_DIR / "checkpoints" / config["algo"]["experiment_name"]).absolute()))
    event_callback = EveryNTimesteps(n_steps=int(config["algo"]["save_checkpoint_every_n_timesteps"]), callback=checkpoint_on_event)

    algo_ppo.learn(
        total_timesteps=int(config["algo"]["train_steps"]), 
        callback=[
            eval_callback,
            event_callback,
            WandbCallback(
                model_save_freq=int(config["algo"]["save_checkpoint_every_n_timesteps"]),
                model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / config["algo"]["experiment_name"] / wandb_run.id).absolute()),
                verbose=2,
            )
        ]
    )

    return sb3_logger, vec_env, eval_env_in_callback


# python scripts_train/train_with_ppo.py config_file=configs/test.yaml
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
