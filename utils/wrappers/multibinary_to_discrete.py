import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultiBinaryToDiscreteWrapper(gym.Wrapper):
    """
    将MultiBinary动作空间转换为Discrete动作空间的Wrapper
    
    MultiBinary(n)有2^n种可能的组合，我们将每种组合映射到0到2^n-1的整数
    """
    def __init__(self, env):
        super().__init__(env)
        
        # 确保原始环境的动作空间是MultiBinary
        assert isinstance(env.action_space, spaces.MultiBinary), \
            "原始环境的动作空间必须是MultiBinary类型"
        
        # 获取二进制动作的维度
        self.n = env.action_space.n
        
        # 计算可能的动作总数 (2^n)
        self.num_actions = (2 **self.n) - 1
        
        # 定义新的Discrete动作空间
        self.action_space = spaces.Discrete(n=self.num_actions, start=1)
        
        # 预计算所有可能的二进制动作组合，加速转换
        self._action_mapping = []
        for i in range(0, self.num_actions + 1):
            # 将整数转换为二进制表示，长度为n，高位补0
            binary = np.array([(i >> k) & 1 for k in range(self.n-1, -1, -1)], dtype=np.int8)
            self._action_mapping.append(binary)

    def step(self, action):
        # 将Discrete动作转换为原始的MultiBinary动作
        binary_action = self._action_mapping[action]
        return self.env.step(binary_action)
