import random, numpy, argparse
import gymnasium as gym
import sys
sys.path.append('../gimitest')
from env_decorator import EnvDecorator
from gtest import GTest
from glogger import GLogger
from gtest_decorator import GTestDecorator
from ganalysis import GAnalyse
import random


class GoalTester(GTest):

    def __init__(self, env):
        super().__init__(env)
        self.goal_counter = 0

    def post_step_test(self, state, action, next_state, reward, terminated, truncated, info, agent_selection):
        # Checks at every step (after the step was executed), if the agent reached the goal (just check if reward +1)

        # Test if reward == 1
        if reward == 1:
            self.goal_counter += 1

        # Store result into episode_data for logger
        self.episode_data["goal_counter"] = self.goal_counter

        return state, action, next_state, reward, terminated, truncated, info

    def pre_reset_configuration(self):
        # Change speed1 in the internal environment variable "channels"
        channels = self.get_attribute(env, "channels")

        channels["speed1"] = random.randint(1,6)
        self.episode_data["speed1"] = channels["speed1"]

        channels["speed2"] = random.randint(1,6)
        self.episode_data["speed2"] = channels["speed2"]

        self.set_attribute(env, "channels", channels)

    def post_reset_test(self):
        # Reset goal counter for new episode
        self.goal_counter = 0



NUM_STEPS = 5002 # Default termination after 2500 frames
env = gym.make('MinAtar/Freeway-v1')

m_gtest = GoalTester(env)
EnvDecorator.decorate(env, m_gtest)

m_logger = GLogger("minitar_freeway")
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
g_analyse = GAnalyse(m_logger)
df = g_analyse.create_episode_dataset(["speed1", "speed2", "goal_counter"])
print(df.head())

# Delete the database of the logger
m_logger.delete_database()