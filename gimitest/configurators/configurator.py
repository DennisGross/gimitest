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

    def set_attribute(self, env, attribute_name, n_value):
        """Sets the value of attribute attribute_name of the environment.
        
        Args:
            env (object): The gym environment object whose attribute needs to be modified.
            attribute_name (str): The name of the attribute to modify.
            n_value (object): The new value to set.

        Returns:
            None: Modifies the attribute of the environment in place.
        
        Raises:
            ValueError: If attribute name does not exist in the environment.
        """
        is_attr = hasattr(env, attribute_name)
        if is_attr:
            setattr(env, attribute_name, n_value)
        else:
            raise AttributeError(f"Attribute {attribute_name} does not exist in environment.")

    def get_attribute(self, env, attribute_name):
        """Fetches the current attribute value of the environment.
        
        Args:
            env (object): The gym environment object whose state needs to be fetched.

        Returns:
            object: The attribute value of the environment.

        Raises:
            ValueError: If attribute name does not exist in the environment.
        """
        is_attr = hasattr(env, attribute_name)
        if is_attr:
            return getattr(env, attribute_name)
        else:
            raise AttributeError(f"Attribute {attribute_name} does not exist in environment.")

    def configure(self, env, test_case_messages):
        """Configures the gym environment. Intended for overriding by subclasses.
        
        Args:
            env (object): The gym environment to configure.
            test_case_messages (list): A list of messages from test cases that may be used for configuration.

        Returns:
            None: Placeholder for child classes to implement custom configuration logic.
        """
        #print("I am the configure method of the Configurator class (modify me).")
        return self.get_attribute(env, self.parameters["state_variable_name"])
