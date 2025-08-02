from pathlib import Path
from copy import deepcopy
import numpy as np
import gymnasium as gym
import optuna

from algorithms.optuna_minimize import SystemIdentificationWithOptuna
from algorithms.utils.consts import HELPER_ENV_CLASS_DICT

PROJECT_ROOT_DIR = Path(__file__).parent.parent


class SystemIdentificationEnv(gym.Env):
    """A custom environment for system identification tasks.
    This environment simulates a system where the dynamics can be adjusted
    based on parameters provided during initialization.
    """

    def __init__(
        self, 
        dynamics_env_id: str,
        params_config: dict,
        obs_real_file_path: str,
        act_real_file_path: str,
        next_obs_real_file_path: str,
        bo_optimizer_n_trials: int = 1000,
        bo_optimizer_n_jobs: int = 4,
        bo_optimizer_sample_num_in_optimize: int = 1000,
        reward_b: float = 1.0,
        max_steps: int = 30,
        loss_threshold: float = 1e-10,
    ):
        """params_config提供了参数的范围，action是选择一个参数，reward是optimizer在数据上优化该参数后的loss。

        Args:
            params (dict): {
                "param1": {
                    "initial_value": 5.0,
                    "range": [param1_low, param1_high],
                    "optimize": true,  # 是否需要优化
                }
            }
        """

        # Prepare the dynamics
        self.env_id = dynamics_env_id
        self.helper_env_class = HELPER_ENV_CLASS_DICT.get(dynamics_env_id, None)
        assert self.helper_env_class is not None, f"env_id {dynamics_env_id} not found in HELPER_ENV_CLASS_DICT: {HELPER_ENV_CLASS_DICT.keys()}"

        # Prepare the configuration of the dynamics
        self.params_config_all = params_config
        self.params_config_to_be_optimized = {
            ky: self.params_config_all[ky] for ky in self.params_config_all.keys() if self.params_config_all[ky].get("optimize", False)
        }
        self.key_list = list(self.params_config_all.keys())
        self.key_list_of_params_to_be_optimized = list(self.params_config_to_be_optimized.keys())
        self.initial_params_all = {
            ky: self.params_config_all[ky]["initial_value"] for ky in self.key_list
        }
        self.initial_params_to_be_optimized = {
            ky: self.params_config_all[ky]["initial_value"] for ky in self.key_list_of_params_to_be_optimized
        }
        self.current_params_to_be_optimized = None
        assert all(a == b for a, b in zip(self.initial_params_all.keys(), self.params_config_all.keys())), f"current_params keys {self.initial_params_all.keys()} must match params_config keys {self.params_config_all.keys()}"

        # Load real observation and action data
        assert Path(PROJECT_ROOT_DIR / obs_real_file_path).exists(), f"Observation real file {PROJECT_ROOT_DIR / obs_real_file_path} does not exist."
        assert Path(PROJECT_ROOT_DIR / act_real_file_path).exists(), f"Action real file {PROJECT_ROOT_DIR / act_real_file_path} does not exist."
        assert Path(PROJECT_ROOT_DIR / next_obs_real_file_path).exists(), f"Next observation real file {PROJECT_ROOT_DIR / next_obs_real_file_path} does not exist."
        self.obs_real = np.load(PROJECT_ROOT_DIR / obs_real_file_path, allow_pickle=True)
        self.act_real = np.load(PROJECT_ROOT_DIR / act_real_file_path, allow_pickle=True)
        self.next_obs_real = np.load(PROJECT_ROOT_DIR / next_obs_real_file_path, allow_pickle=True)

        self.obs_real_cur_episode: np.ndarray = None
        self.act_real_cur_episode: np.ndarray = None
        self.next_obs_real_cur_episode: np.ndarray = None

        # define observation and action spaces
        # 只把需要优化的参数作为observation以及action！！！
        tmp_low = np.array([self.params_config_all[ky]["range"][0] for ky in self.key_list_of_params_to_be_optimized])
        tmp_high = np.array([self.params_config_all[ky]["range"][1] for ky in self.key_list_of_params_to_be_optimized])
        
        self.observation_space = gym.spaces.Box(low=tmp_low, high=tmp_high, dtype=np.float32)
        self.action_space = gym.spaces.MultiBinary(n=len(self.key_list_of_params_to_be_optimized))

        # Initialize the Bayesian optimizer
        self.bo_optimizer_n_trials = bo_optimizer_n_trials
        self.bo_optimizer_n_jobs = bo_optimizer_n_jobs
        self.bo_optimizer_sample_num_in_optimize = bo_optimizer_sample_num_in_optimize
        self.bo_optimizer = SystemIdentificationWithOptuna(
            current_params=self.initial_params_all,
            params_config=self.params_config_to_be_optimized,
            helper_env_class=self.helper_env_class,
        )

        assert len(self.act_real) > self.bo_optimizer_sample_num_in_optimize, f"bo_optimizer_sample_num_in_optimize: {self.bo_optimizer_sample_num_in_optimize} must be smaller than obs/act/next_obs num: {len(self.act_real)}!"

        # env config
        self.reward_b = reward_b
        self.max_steps = max_steps
        self.loss_threshold = loss_threshold

        # Additional info
        self.step_cnt = 0
        self.loss_list = []


    def step(self, action):
        # Simulate the next state based on the current parameters and action
        
        if np.sum(action) == 0:
            next_state = self.to_obs(self.current_params_to_be_optimized)
            cur_step_loss = self.loss_list[-1]
            reward = self.compute_reward(current_step_loss=cur_step_loss)
            self.step_cnt += 1
            self.loss_list.append(cur_step_loss)
            terminated, truncated, success = self.compute_terminated(cur_step_loss)
            info = {
                "params_before_optimize": deepcopy(self.current_params_to_be_optimized),
                "params_selected_to_optimize": deepcopy({}),
                "params_after_optimize": deepcopy(self.current_params_to_be_optimized),
                "loss": cur_step_loss,
                "success": True if success else False,
            }
            return next_state, reward, terminated, truncated, info

        # 1.构造要优化的配置dict
        params_to_be_optimized = self.get_params_to_be_optimized_from_action(self.params_config_to_be_optimized, action)

        # 2.使用Bayesian Optimization优化选中的参数
        self.bo_optimizer.current_params = deepcopy(self.initial_params_all)
        self.bo_optimizer.current_params.update(self.current_params_to_be_optimized)
        self.bo_optimizer.params_config = params_to_be_optimized

        study: optuna.Study = self.bo_optimizer.optimize(
            obs_real=self.obs_real_cur_episode,
            act_real=self.act_real_cur_episode,
            next_obs_real=self.next_obs_real_cur_episode,
            n_trials=self.bo_optimizer_n_trials,
            n_jobs=self.bo_optimizer_n_jobs,
            seed=self.np_random.integers(0, 1000000000),
            show_progress_bar=False,
        )

        bo_optimized_params: dict = study.best_params
        loss_of_bo_optimized_params: float = study.best_value

        # 3.计算next_state
        self.current_params_to_be_optimized.update(bo_optimized_params)
        next_state = self.to_obs(self.current_params_to_be_optimized)

        # 4.计算reward
        reward = self.compute_reward(current_step_loss=loss_of_bo_optimized_params)

        # 5.记录辅助信息
        self.step_cnt += 1
        self.loss_list.append(loss_of_bo_optimized_params)

        # 6.判断terminated和truncated
        terminated, truncated, success = self.compute_terminated(loss_of_bo_optimized_params)

        # 7.准备info
        info = {
            "params_before_optimize": deepcopy(self.bo_optimizer.current_params),
            "params_selected_to_optimize": deepcopy(self.bo_optimizer.params_config),
            "params_after_optimize": deepcopy(bo_optimized_params),
            "loss": loss_of_bo_optimized_params,
            "success": True if success else False,
        }

        return next_state, reward, terminated, truncated, info

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.step_cnt = 0
        self.loss_list = []

        sample_indexes = self.np_random.choice(len(self.act_real), self.bo_optimizer_sample_num_in_optimize, replace=False)
        self.obs_real_cur_episode = self.obs_real[sample_indexes]
        self.act_real_cur_episode = self.act_real[sample_indexes]
        self.next_obs_real_cur_episode = self.next_obs_real[sample_indexes]
        
        self.current_params_to_be_optimized = deepcopy(self.initial_params_to_be_optimized)

        initial_loss = self.bo_optimizer.calc_loss(
            current_params=self.initial_params_all,
            obs_real=self.obs_real_cur_episode,
            act_real=self.act_real_cur_episode,
            next_obs_real=self.next_obs_real_cur_episode,
        )
        self.loss_list.append(initial_loss)

        return self.to_obs(self.current_params_to_be_optimized), {
            "loss": initial_loss,
        }

    def to_obs(self, config: dict) -> np.ndarray:
        """Convert the configuration dictionary to an observation array."""
        return np.array([config[ky] for ky in self.key_list_of_params_to_be_optimized], dtype=np.float32)

    def get_params_to_be_optimized_from_action(self, params, action) -> dict:
        """Extract the parameters to be optimized based on the action taken."""
        assert self.action_space.contains(action), f"Action {action} is not valid for the action space {self.action_space}"

        return {ky: params[ky] for ky, act in zip(self.key_list_of_params_to_be_optimized, action) if act == 1}
    
    def compute_reward(self, current_step_loss: float) -> float:
        dis = np.abs(current_step_loss / self.loss_list[0])
        return - np.power(dis, self.reward_b)

    def compute_terminated(self, current_loss: float) -> tuple[bool, bool, bool]:
        terminated, truncated, success = False, False, False
        
        # 超过最大步长
        if self.step_cnt >= self.max_steps:
            terminated = True
        
        # 达到loss要求
        if current_loss < self.loss_threshold:
            terminated = True
            success = True
        
        return terminated, truncated, success

    def _simulate_dynamics(self, action):
        # Placeholder for actual dynamics simulation logic
        return np.random.rand(*self.observation_space.shape) + action * self.params_config_all.get('power', 1.0)

    # @property
    # def params_config_to_be_optimized(self):
    #     """Get the parameters configuration that are set to be optimized."""
    #     return {
    #         ky: self.params_config_all[ky] for ky in self.params_config_all.keys() if self.params_config_all[ky].get("optimize", False)
    #     }

    # @property
    # def key_list(self):
    #     """Get the list of all parameter keys."""
    #     return list(self.params_config_all.keys())
    
    # @property
    # def key_list_of_params_to_be_optimized(self):
    #     """Get the list of keys for parameters that are set to be optimized."""
    #     return list(ky for ky in self.key_list if self.params_config_all[ky].get("optimize", False))
    
    # @property
    # def initial_params_all(self):
    #     """Get the initial parameters for all keys."""
    #     return {
    #         ky: self.params_config_all[ky]["initial_value"] for ky in self.key_list
    #     }
    
    # @property
    # def initial_params_to_be_optimized(self):
    #     """Get the initial parameters for keys that are set to be optimized."""
    #     return {
    #         ky: self.params_config_all[ky]["initial_value"] for ky in self.key_list_of_params_to_be_optimized
    #     }