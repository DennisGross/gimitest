"""
Configurator Class

Author: Dennis Gross
Email: dennis@simula.no
Date: 16.10.2023
Version: 1.0

Description:
    This Python script provides a Configurator class designed to configure gym environments.
    The class contains methods for modifying and getting the state of the environment as well
    as a method, `configure`, intended for overriding to specify custom configurations based on
    test case messages or other criteria.

Dependencies:
    None

Usage:
    To use this class, instantiate a Configurator object and call its methods to interact with
    a gym environment. To customize behavior, subclass Configurator and override the `configure` method.

Example:
    class CustomConfigurator(Configurator):
        def configure(self, env, test_case_messages):
            # Custom code for configuration
"""

class Configurator():
    """Configurator base class for configuring gym environments.
    
    Attributes:
        parameters (dict): Dictionary of custom parameters for the configurator. Expected to contain a key "state_variable_name".
    """

    def __init__(self, parameters):
        """Initializes the Configurator object with given parameters.

        Args:
            parameters (dict): Custom parameters for the configurator. 
                               Expected to contain a key "state_variable_name" to indicate the state variable.
        """
        self.parameters = parameters

    def modify_state(self, env, n_state):
        """Modifies the state of the environment based on the parameters.
        
        Args:
            env (object): The gym environment object whose state needs to be modified.
            n_state (object): The new state to set.

        Returns:
            None: Modifies the state of the environment in place.
        
        Raises:
            ValueError: If no state variable name is provided in parameters.
        """
        state_variable_name = self.parameters.get("state_variable_name", None)
        if state_variable_name:
            setattr(env, state_variable_name, n_state)
        else:
            raise ValueError("No state variable name provided in parameters (modify me).")

    def get_state(self, env):
        """Fetches the current state of the environment based on the parameters.
        
        Args:
            env (object): The gym environment object whose state needs to be fetched.

        Returns:
            object: The state of the environment.

        Raises:
            ValueError: If no state variable name is provided in parameters.
        """
        state_variable_name = self.parameters.get("state_variable_name", None)
        if state_variable_name:
            return getattr(env, state_variable_name)
        else:
            raise ValueError("No state variable name provided in parameters (modify me).")

    def configure(self, env, test_case_messages):
        """Configures the gym environment. Intended for overriding by subclasses.
        
        Args:
            env (object): The gym environment to configure.
            test_case_messages (list): A list of messages from test cases that may be used for configuration.

        Returns:
            None: Placeholder for child classes to implement custom configuration logic.
        """
        print("I am the configure method of the Configurator class (modify me).")
        return self.get_state(env)
