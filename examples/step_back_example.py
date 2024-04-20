import random, numpy, argparse
import gymnasium as gym
import sys
sys.path.append('../gimitest')
from env_decorator import EnvDecorator
from gtest import GTest
from glogger import GLogger
from gtest_decorator import GTestDecorator
import random

# CMake sure your environment has a step_back method!
class SimpleEnv(gym.Env):
    def __init__(self):
        self.state = 0
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(1)
    def reset(self):
        self.state = 0
        return self.state
    
    def step_back(self):
        self.state -= 1
        return self.state
    
    def step(self, action):
        self.state += 1
        if self.state >= 10:
            return self.state, 0, True, {}
        return self.state, 0, False, {}
    
env = SimpleEnv()
m_gtest = GTest(env)
m_glogger = GLogger("step_back_example")
m_gtest = GTestDecorator.decorate_with_logger(m_gtest, m_glogger)

# Decorate the GTest with the GTestDecorator
EnvDecorator.decorate(env, m_gtest)
env.reset()
print("Before step")
print(env.state)
print(m_gtest.env.unwrapped.state)
env.step(0)
print("After step")
print(env.state)
print(m_gtest.env.unwrapped.state)
m_gtest.step_back()
print("After step back")
print(env.state)
print(m_gtest.env.unwrapped.state)
