import gymnasium as gym
import sys
sys.path.append('../gimitest')
from gym_decorator import GymDecorator
from test_cases.test_case import TestCase
from test_cases.loggers.test_logger import TestLogger
from test_cases.test_case_decorator import TestCaseDecorator
from configurators.configurator import Configurator
from test_cases.analysis.analysis import TestAnalyse
# Init Gym
env = gym.make('CartPole-v1')
# Test Case
m_logger = TestLogger("test_result")
m_test_case = TestCase()
m_test_case = TestCaseDecorator.decorate_test_case_with_test_result(m_test_case, m_logger)
# List of Test Cases
test_cases = [m_test_case]
# Configurator
configurator = Configurator({"state_variable_name": "state"})
# Decorate the environment by extending its reset and step function
env = GymDecorator.decorate_gym(env, test_cases, configurator)
# Run the environment
state, info = env.reset()

for episode in range(10):
    done = truncated = False
    while done == False and truncated == False:
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        if done or truncated:
            state, info = env.reset()

# Analyze the test logs
m_analytics = TestAnalyse(m_logger)
m_analytics.plot_key_value_over_episodes("collected_reward")