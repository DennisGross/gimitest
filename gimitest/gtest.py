import gymnasium as gym
import numpy as np
from queue import Queue

class GTest:

    def __init__(self, env, agents = None, parameters={}):
        self.env = env
        self.agents = agents
        self.episode = -1
        self.step = 0
        self.step_data = {}
        self.episode_data = {}
        self.parameters = parameters

    def pre_step_configuration(self):
        return None

    def pre_step_test(self, agent_selection, action):
        return None
    
    def post_step_configuration(self):
        return None
    
    def post_step_test(self, state, action, next_state, reward, terminated, truncated, info, agent_selection):
        return state, action, next_state, reward, terminated, truncated, info
    
    def pre_reset_configuration(self):
        return None
    
    def pre_reset_test(self):
        return None

    def post_reset_configuration(self, next_state):
        return None
    
    def post_reset_test(self):
        return None

    def step_increment(self):
        self.step += 1

    def episode_increment(self):
        self.episode += 1
        self.step = 0
        self.step_data = {}
        self.episode_data = {}

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
                try:
                    nested_attr = getattr(obj, attr)
                    if isinstance(nested_attr, object) and not isinstance(nested_attr, (str, int, float, bytes)):
                        q.put(nested_attr)
                except:
                    pass
        
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
                try:
                    nested_attr = getattr(obj, attr)
                except:
                    #print("Error", obj, attr)
                    continue
                
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
