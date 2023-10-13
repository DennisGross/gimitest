"""
This module contains the Gimi Gym.
"""
import math
from typing import Tuple, Union
import gym
import argparse
import numpy as np


class GimiGym():

    def __init__(self, env, state_attribute, max_steps):
        self.env = env
        self.state_reference_value, self.state_manipulator = self.__extract_and_modify_attribute_after_steps(env, state_attribute)
        # if state_manipulator is None or not numpy array or int, raise error
        if self.state_manipulator is None or (not isinstance(self.state_reference_value, np.ndarray) and not isinstance(self.state_reference_value, np.int64)):
            if self.state_reference_value is None:
                raise ValueError(f"The specified state attribute does not exist ({state_attribute}).")
            else:
                raise ValueError(f"The specified state attribute is not a numpy attribute ({self.state_reference_value.__class__}).")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.steps = 0
        self.max_steps = max_steps
        print(f"State reference value: {self.state_reference_value}")
        print(f"State manipulator: {self.state_manipulator}")
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")


    def __extract_and_modify_attribute_after_steps(self, env, state_attribute='state'):
        """
        Modify extract the specified attribute from the deepest nested 'env' structure,
        and provide a method to modify it.
        
        :param env: An instance of an OpenAI Gym environment or its subclass.
        :param state_attribute: Name of the attribute to be extracted and modified. Default is 'state'.
        :return: The specified attribute value and a function to modify it.
        """
        env.reset()
        # Recursively delve into the nested 'env' attributes to reach the deepest level
        deepest_env = env
        all_attributes = dir(deepest_env)
        while hasattr(deepest_env, 'env'):
            deepest_env = getattr(deepest_env, 'env')
            all_attributes = dir(deepest_env)

        # Check if the deepest 'env' has an attribute with the name specified in 'state_attribute'
        if hasattr(deepest_env, state_attribute):
            attribute_value = getattr(deepest_env, state_attribute)
            # Define a function to modify the specified attribute
            def modify_attribute(new_value, attr_name=state_attribute, environment=deepest_env):
                setattr(environment, attr_name, new_value)
            return attribute_value, modify_attribute

        all_attributes = dir(deepest_env)
        return None, None

    def reset(self, state=None):
        """
        Resets the environment.
        """
        if state is None:
            state = self.env.reset()
        else:
            self.state_manipulator(state)
            state = self.env.reset()
        state = state[0]
        self.steps = 0
        return state

    def step(self, action):
        """
        Performs a step in the environment.
        """
        next_state, reward, done, truncated, info = self.env.step(action)
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        return next_state, reward, done, info

    def render(self):
        """
        Renders the environment.
        """
        self.env.render()