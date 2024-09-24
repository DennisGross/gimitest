import inspect
import gymnasium as gym
import inspect
import pkgutil
import importlib

def get_environment_source(env):
    """Retrieves the source code of the class for a given gym environment.

    This function checks if the provided object is an instance of gym.Env, unwraps it if possible, and then retrieves and returns the source code of the environment's class.

    Args:
        env (gym.Env): The gym environment instance whose source code is to be retrieved.

    Returns:
        str: The source code of the environment's class, or an empty string if an error occurs.

    Raises:
        TypeError: If the provided object is not a Gym environment.
    """
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
    

def get_environment_attributes(env, types=(object), include_values=False):
    """Retrieves attributes from a gym environment that match specified types,
    optionally including their initial values.

    Args:
        env (gym.Env): An instance of a gym environment from which attributes are to be retrieved.
        types (tuple): A tuple of data types to filter the attributes by. Defaults to (object).
        include_values (bool): If True, returns a dictionary with attribute names and their values; otherwise, returns a list of attribute names.

    Returns:
        dict or list: Depending on 'include_values', a dictionary with attribute names and their values or a list of attribute names.

    Raises:
        TypeError: If the provided object is not a Gym environment.
    """
    # Check if the environment is an instance of a Gym environment
    if not isinstance(env, gym.Env):
        raise TypeError("Provided object is not a Gym environment.")

    try:
        env = env.unwrapped  # Unwrap the environment if possible
    except AttributeError as e:
        print(f"Error unwrapping environment: {e}")

    # Retrieve all attributes of the environment instance
    attributes = dir(env)

    # Initialize a collection to hold the output
    attr_collection = {} if include_values else []

    # Iterate over the attribute names
    for attr in attributes:
        try:
            attr_value = getattr(env, attr)
            # Check if the attribute is not callable and matches one of the specified types
            if not callable(attr_value) and isinstance(attr_value, types):
                if include_values:
                    attr_collection[attr] = attr_value
                else:
                    attr_collection.append(attr)
        except Exception:
            pass  # Ignore exceptions in accessing attributes

    return attr_collection

def get_full_module_code(module, exclude_files=[], include_files=[]):
    """Returns the source code for the specified module and all its submodules,
    optionally excluding or including specific files.

    Args:
        module (Module): The root module from which source code is to be extracted.
        exclude_files (list): A list of substrings that, if present in a module's file path, will exclude it from being included.
        include_files (list): A list of substrings that, if present in a module's file path, ensure its inclusion.

    Returns:
        str: All collected source code from the module and its submodules.
    """
    source_code = ""
    
    def retrieve_module_code(mod):
        """Helper function to retrieve module source and append to the main source_code variable."""
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


def find_function_path(obj, func_name, current_path=""):
    """
    Recursively searches for a function by name in an object and returns the object-path to it.

    :param obj: The object to search within.
    :param func_name: The name of the function to search for.
    :param current_path: The current path in the object (used for recursion).
    :return: The object-path to the function, or None if not found.
    """
    # Check if the object has the function as an attribute
    if hasattr(obj, func_name) and callable(getattr(obj, func_name)):
        return f"{current_path}.{func_name}".strip(".")
    
    # Recursively search the object's attributes
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        # Skip built-in attributes and non-object types
        if not attr_name.startswith('__') and hasattr(attr, '__dict__'):
            result = find_function_path(attr, func_name, f"{current_path}.{attr_name}".strip("."))
            if result:
                return result

    return None