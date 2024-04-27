import inspect
import gymnasium as gym
import inspect
import pkgutil
import importlib

def get_environment_source(env):
    # Check if the environment is an instance of a Gym environment
    if not isinstance(env, gym.Env):
        raise TypeError("Provided object is not a Gym environment.")
    try:
        env = env.unwrapped
    except AttributeError as e:
        print(f"Error unwrapping environment: {e}")
        pass
    try:
        # Get the class of the environment instance
        env_class = env.__class__
        # Retrieve and print the source code of the environment class
        source_code = inspect.getsource(env_class)
        return source_code
    except Exception as e:
        print(f"Error retrieving source code: {e}")
        return ""
    


def get_non_callable_attributes(env):
    # Check if the environment is an instance of a Gym environment
    if not isinstance(env, gym.Env):
        raise TypeError("Provided object is not a Gym environment.")
    try:
        env = env.unwrapped
    except AttributeError as e:
        print(f"Error unwrapping environment: {e}")
        pass
    # Retrieve all attributes and methods of the environment instance
    attributes = dir(env)
    
    # Initialize a list to hold the names of non-callable attributes
    non_callable_attrs = []
    
    # Iterate over the attribute names
    for attr in attributes:
        # Get the attribute value
        try:
            attr_value = getattr(env, attr)
            # Check if the attribute is not callable
            if not callable(attr_value):
                non_callable_attrs.append(attr)
        except Exception as e:
            pass
    
    return non_callable_attrs

def get_int_float_bool_attributes(env):    
    # Check if the environment is an instance of a Gym environment
    if not isinstance(env, gym.Env):
        raise TypeError("Provided object is not a Gym environment.")

    try:
        env = env.unwrapped  # Unwrap the environment if possible
    except AttributeError as e:
        print(f"Error unwrapping environment: {e}")

    # Retrieve all attributes and methods of the environment instance
    attributes = dir(env)
    
    # Initialize a list to hold the names of non-callable attributes that are bool, float, or int
    specific_type_attrs = []
    
    # Iterate over the attribute names
    for attr in attributes:
        try:
            attr_value = getattr(env, attr)
            # Check if the attribute is not callable and is of type bool, float, or int
            if not callable(attr_value) and isinstance(attr_value, (bool, float, int)):
                specific_type_attrs.append(attr)
        except Exception as e:
            pass  # Ignore exceptions in accessing attributes
    
    return specific_type_attrs




def get_full_module_code(module, exclude_files=[], include_files=[]):
    """ Returns the source code for the specified module and all its submodules,
        optionally excluding files specified in exclude_files or including only files in include_files.
    """
    source_code = ""
    
    def retrieve_module_code(mod):
        """ Helper to retrieve module source and append to the main source_code variable. """
        nonlocal source_code
        try:
            module_file = getattr(mod, '__file__', '')
            # Determine inclusion based on include_files if it's not empty
            if include_files:
                if not any(included in module_file for included in include_files):
                    return
            # Check if the current module's file should be excluded
            elif exclude_files:
                if any(excluded in module_file for excluded in exclude_files):
                    return
            
            # Getting the source code of the current module
            source = inspect.getsource(mod)
            source_code += f"\n# Source of {mod.__name__}\n{source}\n"
        except Exception as e:
            # Log errors or handle them as needed
            pass
    
    if module:
        # Starting with the root module
        retrieve_module_code(module)
        
        # Iterating through all submodules
        if hasattr(module, "__path__"):  # Check if the module is a package
            for importer, modname, ispkg in pkgutil.walk_packages(path=module.__path__,
                                                                  prefix=module.__name__ + '.'):
                try:
                    # Importing submodule
                    submodule = importlib.import_module(modname)
                    retrieve_module_code(submodule)
                except Exception as e:
                    # Log errors or handle them as needed
                    pass
    
    return source_code



