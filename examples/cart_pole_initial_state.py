import numpy as np
import gymnasium as gym

import sys
sys.path.append('../gimitest/')

from env_decorator import EnvDecorator
from gtest import GTest


class InitialStateTester(GTest):

    def __init__(self, env, env_seed: int = 0, agent=None):
        super().__init__(env, agent)
        assert env_seed >= 0
        self.env_seed = env_seed
        self.initial_state: np.ndarray = np.zeros(4, dtype=np.float32)


    # def pre_reset_configuration(self):
    #     return {'seed': self.env_seed}


    def post_reset_configuration(self, next_state):
        # deterministic executions by setting the unwrapped environment
        self.set_attribute(self.env.unwrapped, 'state', np.array(self.initial_state, dtype=np.float32))
        return np.array(self.initial_state, dtype=np.float32)


    def set_initial_state(self, state: np.ndarray):
        assert len(state) == 4
        self.initial_state = np.array(self.initial_state, dtype=np.float32)


# exec(open('cart_pole_initial_state.py').read())
MAX_EPISODES = 10
env = gym.make('CartPole-v1')

env_seed = 0
m_gtest = InitialStateTester(env)
EnvDecorator.decorate(env, m_gtest)

# checks deterministic executions without fixing randomness
sequence_list = []
for _ in range(10):
    sequence = [env.reset()[0]]
    for _ in range(100):
        obs, _reward, terminated, _truncated, _info = env.step(0)
        if terminated:
            break
    sequence_list.append(np.vstack(sequence))
assert np.all([np.array_equal(sequence_list[0], s) for s in sequence_list[1:]])