import gymnasium as gym
import sys
sys.path.append('../gimitest')
from gym_decorator import GymDecorator
from test_case import TestCase
from configurator import Configurator


# Init Gym
env = gym.make('CartPole-v1')

# List of Test Cases
test_cases = [TestCase()]
# Configurator
configurator = Configurator({"state_variable_name": "state"})
# Decorate the environment by extending its reset and step function
env = GymDecorator.decorate_gym(env, test_cases, configurator)

# Run the environment
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    if done or truncated:
        observation, info = env.reset()
