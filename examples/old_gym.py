import sys
sys.path.append('../gimitest')
from env_decorator import EnvDecorator
from gtest import GTest
from glogger import GLogger
from gtest_decorator import GTestDecorator
import numpy as np


class OldGym():

    def __init__(self) -> None:
        pass

    def step(self,action):
        return np.array([1]), 1, True, {}
    
    def reset(self):
        return np.array([1])




class RandomAngleTester(GTest):

    def __init__(self, env, agent=None):
        super().__init__(env, agent)
        
    def post_step_test(self, state, action, next_state, reward, terminated, truncated, info, agent_selection):
        # Checks at every step (after the step was executed), if the agent reached the goal (just check if reward +1)
        print(reward)
        return state, action, next_state, reward, terminated, truncated, info



MAX_EPISODES = 10

env = OldGym()
m_gtest = RandomAngleTester(env)
m_logger = GLogger("old_gym")
GTestDecorator.decorate_with_logger(m_gtest, m_logger)
EnvDecorator.decorate(env, m_gtest)
env.reset()

env.step(1)

