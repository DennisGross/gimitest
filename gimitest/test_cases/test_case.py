"""
TestCase Class

Author: Dennis Gross
Email: dennis@simula.no
Date: 16.10.2023
Version: 1.0

Description:
    This Python script provides a TestCase class, designed to facilitate the testing of gym environments.
    The class contains methods that can be overridden to specify the behavior at different stages of an episode
    (e.g., during each step or at the end of an episode).

Dependencies:
    None

Usage:
    To use this script, create a subclass of TestCase and override the methods to specify custom behavior for
    test cases.

Example:
    class CustomTestCase(TestCase):
        def step_execute(self, state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info):
            # Custom code here
"""

class TestCase():
    """Base class for creating test cases for gym environments.
    
    Attributes:
        parameters (dict): Custom parameters for the test case.
    """

    def __init__(self, parameters={}):
        """Initializes the TestCase object with the given parameters.

        Args:
            parameters (dict): Custom parameters for the test case.
        """
        self.parameters = parameters
        self.meta_data = {}
        self.step_data = {}
        self.episode = 0
        self.steps = 0

    def pre_step_execute(self, env, agent_selection, action):
        """Method to be executed before each step of the environment.
        
        Args:
            env (object): The gym environment object.
            agent_selection (object): The agent selection object.

        Returns:
            None: Placeholder for child classes to implement this method.
        """
        #print("I am the pre_step_execute method of the TestCase class (modify me).")
        pass

    def step_execute(self, env, original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info, agent_selection):
        """Method to be executed during each step of the environment.
        
        Args:
            env (object): The gym environment object.
            original_state (object): The current state of the environment.
            action_args (tuple): The arguments for the action taken.
            original_next_state (object): The next state returned by the original step function.
            original_reward (float): The reward returned by the original step function.
            original_terminated (bool): The termination flag returned by the original step function.
            original_truncated (bool): The truncation flag returned by the original step function.
            original_info (dict): The info dictionary returned by the original step function.

        Returns:
            None: Placeholder for child classes to implement this method.
        """
        #print("I am the step_execute method of the TestCase class (modify me).")
        return original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info


    def episode_execute(self):
        """Method to be executed at the start of each episode.
        
        Returns:
            None: Placeholder for child classes to implement this method.
        """
        #print("I am the episode_execute method of the TestCase class (modify me).")
        pass

    def post_episode_execute(self):
        """Method to be executed at the end of each episode.
        
        Returns:
            None: Placeholder for child classes to implement this method.
        """
        #print("I am the post_episode_execute method of the TestCase class (modify me).")
        pass


    def create_message(self):
        """Method for creating messages or information to be passed along.
        
        Returns:
            dict: The message to be passed along.
        """
        #print("I am the get_message method of the TestCase class (modify me).")
        return {}


    def get_message(self, msg):
        """
        Method for getting messages from the configurator.
        """
        # Extend meta_data dictionary with all key, value pairs from msg
        self.meta_data["configurator"] = msg

    def get_pre_reset_message(self, msg):
        """
        Method for getting messages from the configurator.
        """
        # Extend meta_data dictionary with all key, value pairs from msg
        self.meta_data["configurator"] = msg

    def episode_increment(self):
        """
        Method for incrementing the episode number.
        """
        self.episode += 1
        
    def step_increment(self):
        """
        Method for incrementing the step number.
        """
        self.steps += 1
