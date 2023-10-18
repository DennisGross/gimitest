import gymnasium as gym
import sys
sys.path.append('../gimitest')
from gym_decorator import GymDecorator
from test_cases.test_case import TestCase
from test_cases.results.test_result import TestResult
from test_cases.test_case_decorator import TestCaseDecorator
from configurators.configurator import Configurator

# Init Gym
env = gym.make('CartPole-v1')
# Test Case
m_test_case = TestCase()
m_test_case = TestCaseDecorator.decorate_test_case_with_test_result(m_test_case, TestResult("test_result"))
# List of Test Cases
test_cases = [m_test_case]
# Configurator
configurator = Configurator({"state_variable_name": "state"})
# Decorate the environment by extending its reset and step function
env = GymDecorator.decorate_gym(env, test_cases, configurator)
# Run the environment
state, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    if done or truncated:
        state, info = env.reset()