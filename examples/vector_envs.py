import gymnasium as gym
import numpy as np
import random
import sys
sys.path.append('../gimitest')
from env_decorator import EnvDecorator
from gtest import GTest
from glogger import GLogger
from gtest_decorator import GTestDecorator

class RandomAngleTester(GTest):

    def __init__(self, env, agent=None):
        super().__init__(env, agent)

    def post_step_test(self, state, action, next_state, reward, terminated, truncated, info, agent_selection):
        print(self.env.__class__.__name__)
        print(len(self.env.action_space))
        return state, action, next_state, reward, terminated, truncated, info



MAX_EPISODES = 3
MAX_STEPS = 10

envs = gym.vector.make("CartPole-v1", num_envs=3)

m_gtest = RandomAngleTester(envs)
EnvDecorator.decorate(envs, m_gtest)


m_logger = GLogger("lander_log")
GTestDecorator.decorate_with_logger(m_gtest, m_logger)

for episode_idx in range(MAX_EPISODES):
    states, infos = envs.reset(seed=42)
    episode_reward = 0
    for i in range(MAX_STEPS):
        print(states)
        print(infos)
        actions = np.array(envs.action_space.sample())
        print(actions)
        observations, rewards, termination, truncation, infos = envs.step(actions)
        print(observations)
        print(rewards)
        print(termination)
        print(truncation)
        print(infos)
        episode_reward += np.mean(rewards)
    print(f"{episode_idx} Episode Reward: {episode_reward}")
        
    

