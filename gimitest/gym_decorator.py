"""
GymDecorator Class

Author: Dennis Gross
Email: dennis@simula.no
Date: 16.10.2023
Version: 1.0

Description:
    This Python script provides a decorator class for gym environments. 
    The GymDecorator class aims to extend gym's functionalities by adding a layer for test cases and configurability.
    This class may be particularly useful for adding custom behavior to gym environments without altering the original code.

Dependencies:
    - gymnasium (as gym)

Usage:
    To use this script, call the `decorate_gym` static method with a gym environment, test cases, and a configurator.
    The returned environment will have extended functionalities.

Example:
    decorated_env = GymDecorator.decorate_gym(env, test_cases, configurator)
"""

import gymnasium as gym  # Importing gymnasium as gym to work as the base for the decorator

class GymDecorator:
    """Decorator class for gym environments to add additional functionalities like test cases and configurability."""

    @staticmethod
    def decorate_gym(env, test_cases, configurator):
        """Decorates a gym environment with additional functionalities.
        
        Args:
            env (object): The gym environment to decorate.
            test_cases (list): List of test cases to add.
            configurator (object): Configurator object for initial state configuration.

        Returns:
            object: The decorated gym environment.
        """
        env.tmp_storage_of_state = None
        env.step = GymDecorator.__decorate_step_function(env, env.step, test_cases, configurator)
        env.reset = GymDecorator.__decorate_reset_function(env, env.reset, test_cases, configurator)
        return env

    @staticmethod
    def __decorate_step_function(env, original_step_function, test_cases, configurator=None):
        """Internal method to decorate the step function of a gym environment.
        
        Args:
            env (object): The gym environment to decorate.
            original_step_function (function): The original step function.
            test_cases (list): List of test cases to add.

        Returns:
            function: The decorated step function.
        """
        def wrapper(*action_args, **kwargs):
            if configurator != None:
                # Active Testing with old test messages
                configurator.active_configuration_pre_step(env)
            # Call the original step function
            original_next_state, original_reward, original_terminated, original_truncated, original_info = original_step_function(*action_args, **kwargs)
            if configurator != None:
                # Active Testing with old test messages
                configurator.active_configuration_post_step(env)
            
            # Handle test cases if any
            if test_cases is None:
                env.tmp_storage_of_state = original_next_state
                return original_next_state, original_reward, original_terminated, original_truncated, original_info
            else:
                tmp_next_state = original_next_state
                tmp_reward = original_reward
                tmp_terminated = original_terminated
                tmp_truncated = original_truncated
                tmp_info = original_info
                for test_case in test_cases:
                    tmp_state, tmp_action_args, tmp_next_state, tmp_reward, tmp_terminated, tmp_truncated, tmp_info = test_case.step_execute(env, env.tmp_storage_of_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info)
                env.tmp_storage_of_state = original_next_state
                return tmp_next_state, tmp_reward, tmp_terminated, tmp_truncated, tmp_info
        return wrapper

    @staticmethod
    def __decorate_reset_function(env, original_reset_function, test_cases, configurator):
        """Internal method to decorate the reset function of a gym environment.
        The goal of this function is to use the test cases IN THE END OF THE EPISODE and then use its results to configure the initial state of the next episode via the configurator.
        The configurator has then the possibility to broadcast an message to all test cases.
        
        Args:
            env (object): The gym environment to decorate.
            original_reset_function (function): The original reset function.
            test_cases (list): List of test cases to add.
            configurator (object): Configurator object for initial state configuration.

        Returns:
            function: The decorated reset function.
        """
        def wrapper(*args, **kwargs):
            test_case_messages = []
            # Handle test cases if any
            if test_cases is not None:
                for test_case in test_cases:
                    test_case.episode_execute()
                    test_case_messages.append(test_case.create_message())

            # Pre reset configure
            if configurator is not None:
                # Get already the messages from the test cases
                more_args = configurator.configuration_pre_reset(env, test_case_messages)
                # Check if more_args is instance
                if isinstance(more_args, dict)==False:
                    raise TypeError("The pre_reset_configure method of the configurator must return a dictionary.")
                # update kwars with more_args
                kwargs.update(more_args)
                if test_cases is not None:
                    for test_case in test_cases:
                        # Get the message from the configurator for all test cases
                        test_case.get_pre_reset_message(configurator.create_pre_reset_message())

            # Call the original reset function
            next_state, info = original_reset_function(*args, **kwargs)
            
            env.tmp_storage_of_state = next_state

            # Apply configurator if set
            if configurator is not None:
                # Get same test case messages as before but after reset
                tmp_state = configurator.configuration_post_reset(env, test_case_messages)
                if tmp_state is not None:
                    env.tmp_storage_of_state = tmp_state
                if test_cases is not None:
                    for test_case in test_cases:
                        test_case.get_message(configurator.create_post_reset_message())

            

            return env.tmp_storage_of_state, info
        return wrapper

    
