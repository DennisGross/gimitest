import random, numpy, argparse
import gymnasium as gym
import sys
sys.path.append('../gimitest')
from env_decorator import EnvDecorator
from gtest import GTest
from glogger import GLogger
from gtest_decorator import GTestDecorator
import random
import Box2D





class RandomAngleTester(GTest):

    def __init__(self, env, agent=None):
        super().__init__(env, agent)
        




    



MAX_EPISODES = 10
env = gym.make('LunarLander-v2')    # pip install gymnasium[box2d]

m_gtest = RandomAngleTester(env)
EnvDecorator.decorate(env, m_gtest)




print(m_gtest.get_module_attribute("FPS"))
m_gtest.set_module_attribute("FPS", 60)
print(m_gtest.get_module_attribute("FPS"))
exit(0)
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
    rewards.append(episode_reward)
    
    print(f"{episode_idx} Episode Reward: {episode_reward}")

print(f"Average reward: {numpy.mean(rewards)}")
# Create dataset
df = m_logger.create_episode_dataset(["angle", "collected_reward"])
print(df.head(n=100))
df = m_logger.create_step_dataset()
print(df)
# Delete the database of the logger
#m_logger.delete_database()
