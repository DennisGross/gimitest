import gymnasium as gym
import sys
sys.path.append('../gimitest')
from gym_decorator import GymDecorator
from test_cases.test_case import TestCase
from configurators.configurator import Configurator
from test_cases.loggers.test_logger import TestLogger
from test_cases.test_case_decorator import TestCaseDecorator
from test_cases.analysis.analysis import TestAnalyse
import random
class ParameterConfigurator(Configurator):
        


    def configuration_post_reset(self, env, test_case_messages):
        """Configures the gym environment. Intended for overriding by subclasses.
        
        Args:
            env (object): The gym environment to configure.
            test_case_messages (list): A list of messages from test cases that may be used for configuration.

        Returns:
            None: Placeholder for child classes to implement custom configuration logic.
        """
        parameter1 = self.parameters["parameter1"]
        parameter1_lower_bound = self.parameters["parameter1_lower_bound"]
        parameter1_upper_bound = self.parameters["parameter1_upper_bound"]
        parameter1_step = self.parameters["parameter1_step"]
        # Get random value between lower and upper bound with step size
        self.value1 = random.uniform(parameter1_lower_bound, parameter1_upper_bound)
        parameter2 = self.parameters["parameter2"]
        parameter2_lower_bound = self.parameters["parameter2_lower_bound"]
        parameter2_upper_bound = self.parameters["parameter2_upper_bound"]
        parameter2_step = self.parameters["parameter2_step"]
        # Get random value between lower and upper bound with step size
        self.value2 = random.uniform(parameter2_lower_bound, parameter2_upper_bound)
        # Set the environment parameters
        self.set_attribute(env, parameter1, self.value1)
        self.set_attribute(env, parameter2, self.value2)
        print("Parameter 1: ", self.value1)
        print("Parameter 2: ", self.value2)
        # Return the environment state
        return self.get_attribute(env, self.parameters["state_variable_name"])


    def create_pre_reset_message(self):
        """Method for getting messages or information to be passed along.
        
        Returns:
            dict: The message to be passed along.
        """
        # Create a message with the parameters and values
        return {}


    def create_message(self):
        """Method for getting messages or information to be passed along.
        
        Returns:
            dict: The message to be passed along.
        """
        # Create a message with the parameters and values
        message = {"parameter1": self.value1, "parameter2": self.value2}
        return message


# Init Gym
env = gym.make('CartPole-v1')
# Test Case
m_logger = TestLogger("para1_para2_reward")
m_test_case = TestCase()
m_test_case = TestCaseDecorator.decorate_test_case_with_test_logger(m_test_case, m_logger)
# List of Test Cases
test_cases = [m_test_case]
# Configurator
configurator = ParameterConfigurator({"state_variable_name": "state", "parameter1": "masscart", "parameter1_lower_bound": 0, "parameter1_upper_bound": 1.5, "parameter1_step": 0.1, "parameter2": "length", "parameter2_lower_bound": 0.1, "parameter2_upper_bound": 1.0, "parameter2_step": 0.05})
# Decorate the environment by extending its reset and step function
env = GymDecorator.decorate_gym(env, test_cases, configurator)
# Run the environment
state, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    if done or truncated:
        state, info = env.reset()

# Analyze the test logs
m_analytics = TestAnalyse(m_logger)
m_analytics.plot_key_value_over_episodes("collected_reward")
m_analytics.plot_key1_key2_and_value("parameter1", "parameter2", "collected_reward", xlabel="masscart", ylabel="length")

# Delete the test logs
m_logger.delete_test_folder()