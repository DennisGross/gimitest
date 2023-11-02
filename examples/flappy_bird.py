import flappy_bird_gymnasium
import gymnasium
import sys
sys.path.append('../gimitest')
from gym_decorator import GymDecorator
from test_cases.test_case import TestCase
from test_cases.loggers.test_logger import TestLogger
from test_cases.test_case_decorator import TestCaseDecorator
from configurators.configurator import Configurator
from test_cases.analysis.analysis import TestAnalyse
import numpy as np

env = gymnasium.make("FlappyBird-v0", render_mode="human")

# Noisy Sensors
class TestNoisySensors(TestCase):

    def __init__(self, parameters={}):
        super(TestNoisySensors, self).__init__(parameters)

    def step_execute(self, env, original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info):
        perturbation = np.random.normal(-0.1, 0.1, original_next_state.shape)
        # Add noise to the state
        original_next_state += perturbation
        return original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info


m_logger = TestLogger("flappy_bird")
m_test_case = TestNoisySensors()
m_test_case = TestCaseDecorator.decorate_test_case_with_test_logger(m_test_case, m_logger)
test_cases = [m_test_case]
# Decorate gym
env = GymDecorator.decorate_gym(env, test_cases, None)

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()