from copy import deepcopy
import numpy as np
import gymnasium as gym

from stable_baselines3.common.policies import BasePolicy


def collect_samples(policy: BasePolicy, env: gym.Env, num_samples: int, deterministic: bool = True, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect samples from the environment using the given policy.
    
    Args:
        policy (BasePolicy): The policy to use for action selection.
        env (gym.Env): The environment to collect samples from.
        num_samples (int): The number of samples to collect.
        deterministic (bool): Whether to use deterministic action selection.
    
    Returns:
        np.ndarray: Collected observations in the form of an array of observations.
        np.ndarray: Collected actions in the form of an array of actions.
        np.ndarray: Collected next observations in the form of an array of next observations.
    """
    observations = []
    actions = []
    next_observations = []

    obs, _ = env.reset(seed=seed)
    
    for _ in range(num_samples):
        action, _ = policy.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        observations.append(deepcopy(obs))
        actions.append(action)
        next_observations.append(deepcopy(next_obs))

        obs = next_obs
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    return np.array(observations), np.array(actions), np.array(next_observations)
