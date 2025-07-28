import sys
from pathlib import Path
from copy import deepcopy
import numpy as np
import optuna

import gymnasium as gym
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC

PROJECT_ROOT_DIR = Path(__file__).parent.parent

if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

import envs
from envs.utils.env_config_mixins import ConfigurableEnv, DynamicsMixin


class SystemIdentificationWithOptuna:
    """Utilizing Bayesian Optimization for system identification in custom environments.
    """

    def __init__(self, current_params: dict, params_config: dict, helper_env_class: DynamicsMixin):
        """_summary_

        Args:
            params_config (dict): system parameters configurations, form: {
                'param1': {
                    'range': [1, 10],
                    'search_num': 10,  # 将range多少等分
                },
                'param2': {}
            }
        """

        self.current_params = current_params
        self.params_config = params_config
        self.helper_env_class = helper_env_class

    def optimize(self, obs_real: np.ndarray, act_real: np.ndarray, next_obs_real: np.ndarray, n_iter: int=1000, seed: int=42):

        def objective(trial):
            params = {k: trial.suggest_float(k, v["range"][0], v["range"][1]) for k, v in self.params_config.items()}
            tmp_params = deepcopy(self.current_params)
            tmp_params.update(params)
            helper_env = self.helper_env_class(**tmp_params)
            next_obs_sim = np.array([self.helper_env_class.calc_next_obs(state=obs, action=act, helper_env=helper_env) for obs, act in zip(obs_real, act_real)])
            return np.mean((next_obs_sim - next_obs_real)**2)

        study = optuna.create_study(
            direction='minimize', 
            sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_iter, n_jobs=-1)

        return study
