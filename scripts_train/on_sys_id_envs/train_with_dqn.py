import gymnasium as gym
import numpy as np
from pathlib import Path
import torch as th
import sys
from omegaconf import OmegaConf
import optuna

from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.dqn import DQN, MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent

if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

import envs
from envs.sys_id_env import SystemIdentificationEnv
from utils.wrappers.multibinary_to_discrete import MultiBinaryToDiscreteWrapper


optuna.logging.set_verbosity(optuna.logging.WARNING)  # 关闭Optuna控制台的输出
np.seterr(all="raise")  # 检查nan


def train(config, wandb_run):

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / config["rl"]["experiment_name"]).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    # prepare env
    training_env = SystemIdentificationEnv(
        dynamics_env_id=config["dynamics_env"]["id"],
        params_config=config["dynamics_env"]["params_config"],
        obs_real_file_path=PROJECT_ROOT_DIR / config["data_collected_from_real"]["obs_file_path"],
        act_real_file_path=PROJECT_ROOT_DIR / config["data_collected_from_real"]["act_file_path"],
        next_obs_real_file_path=PROJECT_ROOT_DIR / config["data_collected_from_real"]["next_obs_file_path"],
        bo_optimizer_n_trials=config["bo_optimizer"]["n_trials"],
        bo_optimizer_n_jobs=config["bo_optimizer"]["n_jobs"],
        reward_b=config["sys_id_env"]["reward_b"],
        max_steps=config["sys_id_env"]["max_steps"],
        loss_threshold=config["sys_id_env"]["loss_threshold"],
    )
    training_env = MultiBinaryToDiscreteWrapper(training_env)

    eval_env_in_callback = SystemIdentificationEnv(
        dynamics_env_id=config["dynamics_env"]["id"],
        params_config=config["dynamics_env"]["params_config"],
        obs_real_file_path=PROJECT_ROOT_DIR / config["data_collected_from_real"]["obs_file_path"],
        act_real_file_path=PROJECT_ROOT_DIR / config["data_collected_from_real"]["act_file_path"],
        next_obs_real_file_path=PROJECT_ROOT_DIR / config["data_collected_from_real"]["next_obs_file_path"],
        bo_optimizer_n_trials=config["bo_optimizer"]["n_trials"],
        bo_optimizer_n_jobs=config["bo_optimizer"]["n_jobs"],
        reward_b=config["sys_id_env"]["reward_b"],
        max_steps=config["sys_id_env"]["max_steps"],
        loss_threshold=config["sys_id_env"]["loss_threshold"],
    )
    eval_env_in_callback = MultiBinaryToDiscreteWrapper(eval_env_in_callback)

    # DQN hyperparams:
    dqn_algo = DQN(
        MlpPolicy,
        training_env,
        learning_rate=config["rl"]["learning_rate"],
        buffer_size=int(config["rl"]["buffer_size"]),
        learning_starts=int(config["rl"]["learning_starts"]),
        batch_size=int(config["rl"]["batch_size"]),
        tau=0.99,
        gamma=config["rl"]["gamma"],
        train_freq=int(config["rl"]["train_freq"]),
        gradient_steps=int(config["rl"]["gradient_steps"]),
        target_update_interval=int(config["rl"]["target_update_interval"]),
        policy_kwargs=dict(
            # net_arch=config["rl"]["net_arch"],
            net_arch=[128, 128],
            # activation_fn=th.nn.Tanh
        ),
        seed=config["rl"]["seed"],
        device=config["rl"]["device"],
        verbose=1,
    )

    dqn_algo.set_logger(sb3_logger)

    # callback: evaluate, save best
    eval_callback = EvalCallback(
        eval_env_in_callback, 
        best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / config["rl"]["experiment_name"]).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "logs" / config["rl"]["experiment_name"]).absolute()), 
        eval_freq=config["rl"]["eval_freq"],  # 多少次env.step()评估一次，此处设置为1000，因为VecEnv有72个并行环境，所以实际相当于72*1000次step，评估一次
        n_eval_episodes=config["rl"]["n_eval_episodes"],  # 每次评估使用多少条轨迹
        deterministic=True, 
        render=False,
    )

    dqn_algo.learn(
        total_timesteps=int(config["rl"]["train_steps"]),
        callback=[
            eval_callback,
            WandbCallback(
                model_save_freq=config["rl"]["save_checkpoint_every_n_timesteps"],
                model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / config["rl"]["experiment_name"] / wandb_run.id).absolute()),
                verbose=2,
            )
        ]
    )

    # save replay buffer
    dqn_algo.save_replay_buffer(PROJECT_ROOT_DIR / "checkpoints" / config["rl"]["experiment_name"] / "replay_buffer")

    return sb3_logger, training_env, eval_env_in_callback


# python scripts_train/on_sys_id_envs/train_with_dqn.py config_file=configs/sys_id/custom_pendulum/g_9_5_m_0_9_l_1_2/seed_1.yaml 
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
