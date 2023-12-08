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

    def __init__(self, env):
        super().__init__(env)

    def post_reset_configuration(self, next_state):
        # Random angle changer
        angle = random.uniform(-0.5, 0.5)
        # Update environment state
        env = self.env.unwrapped
        print('====================')
        print("Original State")
        print(next_state)
        print('Original Angle', env.lander.angle)
        env.lander.angle = angle
        print('Modified Angle', env.lander.angle)
        # Update next state that is passed to the agent
        next_state[4] = angle
        print('State that is now passed to the agent')
        print(next_state)
        # Store the angle in the episode data for the logger
        self.episode_data["angle"] = angle
        return next_state



NUM_STEPS = 100
env = gym.make('LunarLander-v2')    # pip install gymnasium[box2d]

m_gtest = RandomAngleTester(env)
EnvDecorator.decorate(env, m_gtest)


m_logger = GLogger("lander_log")
GTestDecorator.decorate_with_logger(m_gtest, m_logger)

# Interact with the environment
state, info = env.reset()
for _ in range(NUM_STEPS):
    action = env.action_space.sample()  # Randomly sample an action
    next_state, reward, done, truncated, info = env.step(action)
    #print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
    if done or truncated:
        state, info = env.reset()
    else:
        state = next_state

# Create dataset
df = m_logger.create_episode_dataset(["angle", "collected_reward"])
print(df.head())
df = m_logger.create_step_dataset()
print(df.head())
# Delete the database of the logger
m_logger.delete_database()
