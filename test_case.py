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

    def step_execute(self, state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info):
        """Method to be executed during each step of the environment.
        
        Args:
            state (object): The current state of the environment.
            action_args (tuple): The arguments for the action taken.
            original_next_state (object): The next state returned by the original step function.
            original_reward (float): The reward returned by the original step function.
            original_terminated (bool): The termination flag returned by the original step function.
            original_truncated (bool): The truncation flag returned by the original step function.
            original_info (dict): The info dictionary returned by the original step function.

        Returns:
            None: Placeholder for child classes to implement this method.
        """
        print("I am the step_execute method of the TestCase class (modify me).")
        return original_next_state, original_reward, original_terminated, original_truncated, original_info

    def step_store(self):
        """Method for storing information after each step.
        The storage location can be defined via the parameters.
        Returns:
            None: Placeholder for child classes to implement this method.
        """
        print("I am the step_store method of the TestCase class  (modify me).")

    def step_load(self, path):
        """Method for loading information of a step from a path.

        Args:
            path (str): The path to load the information from.
        """
        print("I am the step_load method of the TestCase class  (modify me).")

    def episode_execute(self):
        """Method to be executed at the start of each episode.
        
        Returns:
            None: Placeholder for child classes to implement this method.
        """
        print("I am the episode_execute method of the TestCase class (modify me).")

    def episode_store(self):
        """Method for storing information at the end of each episode.
        The storage location can be defined via the parameters.
        
        Returns:
            None: Placeholder for child classes to implement this method.
        """
        print("I am the episode_store method of the TestCase class  (modify me).")

    def episode_load(self, path):
        """Method for loading information of a episode from a path.

        Args:
            path (str): The path to load the information from.
        """
        print("I am the episode_load method of the TestCase class  (modify me).")

    def get_message(self):
        """Method for getting messages or information to be passed along.
        
        Returns:
            dict: The message to be passed along.
        """
        print("I am the get_message method of the TestCase class (modify me).")
        return {}
