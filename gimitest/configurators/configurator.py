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
from queue import Queue

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

    def __set_breadth_first_attribute(self, root_obj, attribute_name, n_value):
        """Sets the value of attribute attribute_name of the object using breadth-first search.
        
        Args:
            root_obj (object): The root object whose attribute needs to be modified.
            attribute_name (str): The name of the attribute to modify.
            n_value (object): The new value to set.

        Returns:
            bool: True if the attribute was successfully modified, otherwise False.
        """
        
        q = Queue()
        q.put(root_obj)
        
        while not q.empty():
            obj = q.get()
            
            if hasattr(obj, attribute_name):
                setattr(obj, attribute_name, n_value)
                return True
            
            for attr in dir(obj):
                nested_attr = getattr(obj, attr)
                if isinstance(nested_attr, object) and not isinstance(nested_attr, (str, int, float, bytes)):
                    q.put(nested_attr)
        
        return False

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
        
        is_attr_set = self.__set_breadth_first_attribute(env, attribute_name, n_value)
        
        if not is_attr_set:
            raise AttributeError(f"Attribute {attribute_name} does not exist in environment.")

    def __get_breadth_first_attribute(self, root_obj, attribute_name):
        """Fetches the value of attribute attribute_name of the object using breadth-first search.
        
        Args:
            root_obj (object): The root object whose attribute value needs to be fetched.
            attribute_name (str): The name of the attribute to fetch.

        Returns:
            object: The value of the attribute if found, otherwise None.
        """
        
        q = Queue()
        q.put(root_obj)
        
        while not q.empty():
            obj = q.get()
            
            if hasattr(obj, attribute_name):
                return getattr(obj, attribute_name)
            
            for attr in dir(obj):
                nested_attr = getattr(obj, attr)
                if isinstance(nested_attr, object) and not isinstance(nested_attr, (str, int, float, bytes)):
                    q.put(nested_attr)
                    
        return None  # Return None if attribute is not found

    def get_attribute(self, env, attribute_name):
        """Fetches the current attribute value of the environment.
        
        Args:
            env (object): The gym environment object whose state needs to be fetched.
            attribute_name (str): The name of the attribute to fetch.

        Returns:
            object: The attribute value of the environment.
        
        Raises:
            AttributeError: If attribute name does not exist in the environment.
        """
        
        attr_value = self.__get_breadth_first_attribute(env, attribute_name)
        
        if attr_value is not None:
            return attr_value
        else:
            raise AttributeError(f"Attribute {attribute_name} does not exist in environment.")

    def pre_reset_configure(self, env, test_case_messages):
        """Method for configuring the environment before the reset method is called.
        
        Args:
            env (object): The gym environment to configure.
            test_case_messages (list): A list of messages from test cases that may be used for configuration.

        Returns:
            None: Placeholder for child classes to implement custom configuration logic.
        """
        #print("I am the pre_reset_configure method of the Configurator class (modify me).")
        return {}

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

    def active_configuration(self, env):
        """Method for configuring the environment at each step
        
        Args:
            env (object): The gym environment to configure.
            test_case_messages (list): A list of messages from test cases that may be used for configuration.

        Returns:
            None: Placeholder for child classes to implement custom configuration logic.
        """
        #print("I am the active_configuration method of the Configurator class (modify me).")
        return {}


    def create_message(self):
        """Method for getting messages or information to be passed along.
        
        Returns:
            dict: The message to be passed along.
        """
        #print("I am the get_message method of the TestCase class (modify me).")
        return {}

    def create_pre_reset_message(self):
        """Method for getting messages or information to be passed along.
        
        Returns:
            dict: The message to be passed along.
        """
        #print("I am the get_message method of the TestCase class (modify me).")
        return {}

