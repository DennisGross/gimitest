import gymnasium as gym
from gym_decorator import GymDecorator
from test_case import TestCase
from configurator import Configurator

# Init Gym
env = gym.make('CartPole-v1')

# Test Cases
test_cases = [TestCase()]
configurator = Configurator({"state_variable_name": "state"})
# Extend the step function using the decorator
env = GymDecorator.decorate_gym(env, test_cases, configurator)

# Test the extended environment
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    if done:
        observation, info = env.reset()
