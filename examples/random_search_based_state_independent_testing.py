import random, numpy, argparse
import gymnasium as gym
import sys
sys.path.append('../gimitest')
from env_decorator import EnvDecorator
from gtest import GTest
from glogger import GLogger
from gtest_decorator import GTestDecorator
from testing.random_search_based_state_independent_testing import RandomSearchBasedStateIndependentTesting
import random
import Box2D


MAX_EPISODES = 2
env = gym.make('CartPole-v1')    # pip install gymnasium[box2d]

m_gtest = RandomSearchBasedStateIndependentTesting(env, parameters={"gravity": {"lower_bound": -10.0, "upper_bound": -5.0, "type": "float"}})
EnvDecorator.decorate(env, m_gtest)


m_logger = GLogger("cartpole")
GTestDecorator.decorate_with_logger(m_gtest, m_logger)


rewards = []

for episode_idx in range(MAX_EPISODES):
    state, info = env.reset()
    episode_reward = 0
    done = False
    truncated = False
    steps = 0
    while (not done) and (truncated is False):
        action = env.action_space.sample()  # Randomly sample an action
        next_state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        state = next_state
        break
    rewards.append(episode_reward)
    
    print(f"{episode_idx} Episode Reward: {episode_reward}")
   

m_gtest.clean_up()
print(f"Average reward: {numpy.mean(rewards)}")
# Create dataset
df = m_logger.create_episode_dataset(["gravity", "collected_reward"])
print(df.head(n=100))
df = m_logger.create_step_dataset()
print(df)
# Delete the database of the logger
m_logger.delete_database()
