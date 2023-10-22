import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
sys.path.append('../gimitest')
from gym_decorator import GymDecorator
from test_cases.test_case import TestCase
from configurators.configurator import Configurator
from test_cases.loggers.test_logger import TestLogger
from test_cases.test_case_decorator import TestCaseDecorator
import random

class SeedConfigurator(Configurator):

    def __init__(self, parameters={}):
        super(SeedConfigurator, self).__init__(parameters)

    def pre_reset_configure(self, env, test_case_messages):
        # Set environment seed
        random.seed(self.parameters["seed"])
        np.random.seed(self.parameters["seed"])
        return {"seed": self.parameters["seed"]}



# Example usage
env = gym.make('CartPole-v1')

# Logger and dummy TestCase
m_logger = TestLogger("test_result")
m_test_case = TestCase()
m_test_case = TestCaseDecorator.decorate_test_case_with_test_logger(m_test_case, m_logger)
test_cases = [m_test_case]
# Configurator
configurator = SeedConfigurator({"state_variable_name": "state", "seed": 42})
# Decorate gym
env = GymDecorator.decorate_gym(env,  test_cases, configurator)


# Interact with the environment
state, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Randomly sample an action
    next_state, reward, done, truncated, info = env.step(action)
    print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
    if done or truncated:
        state, info = env.reset()
    else:
        state = next_state
