import sys
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

import gymnasium as gym
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.common.evaluation import evaluate_policy

PROJECT_ROOT_DIR = Path(__file__).parent.parent

if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

import envs


def evaluate(algo_config: OmegaConf, env_config: OmegaConf, eval_conf: OmegaConf):
    env = gym.make(
        env_config["env"]["id"], 
        **env_config["env"]["config"] if "config" in env_config["env"] else {}, 
        render_mode="human" if eval_conf.visualize else None
    )

    # print("env configs: ", env.unwrapped.g, env.unwrapped.m, env.unwrapped.l)

    if eval_conf["algo"] == "ppo":
        model = PPO.load(PROJECT_ROOT_DIR / "checkpoints" / algo_config["algo"]["experiment_name"] / "best_model", env=env)
    elif eval_conf["algo"] == "sac":
        model = SAC.load(PROJECT_ROOT_DIR / "checkpoints" / algo_config["algo"]["experiment_name"] / "best_model", env=env)
    else:
        raise ValueError(f"Unsupported algorithm: {eval_conf['algo']}")
    
    res = evaluate_policy(
        model=model.policy, 
        env=env, 
        n_eval_episodes=eval_conf.n_eval_episodes, 
        render=False, 
        deterministic=True,
        return_episode_rewards=True,
    )

    print(res, np.array(res[0]).mean())

    return res

# python scripts_eval/evaluate.py algo=ppo algo_config=configs/custom_pendulum/g_10_0_m_1_0_l_1_0/ppo/seed_4.yaml env_config=configs/custom_pendulum/g_10_0_m_1_0_l_1_0/ppo/seed_4.yaml n_eval_episodes=5 visualize=true
# python scripts_eval/evaluate.py algo=sac algo_config=configs/pendulum/sac/seed_1.yaml env_config=configs/pendulum/sac/seed_1.yaml n_eval_episodes=100 visualize=false
if __name__ == "__main__":
    conf = OmegaConf.from_cli()
    
    algo_config_dir = conf.algo_config
    env_config_dir = conf.env_config

    algo_config = OmegaConf.load(algo_config_dir)
    env_config = OmegaConf.load(env_config_dir)

    default_conf = OmegaConf.create({
        "n_eval_episodes": 10,
        "visualize": False,
    })

    eval_conf = OmegaConf.merge(default_conf, conf)

    print("Train on:      ", algo_config["env"])

    print("Evaluation on: ", env_config["env"])

    evaluate(algo_config, env_config, eval_conf)
