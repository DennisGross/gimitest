import inspect
import gymnasium as gym
import sys
sys.path.append('../gimitest')
from inspecting.inspect import *


# Example usage:
if __name__ == "__main__":
    # Create an instance of a specific Gym environment
    env = gym.make('LunarLander-v2')
    # Print the source code of the environment
    print("Given")
    print("Environment source code:")
    print(get_environment_source(env))
    print("Non callable attributes:")
    print(get_non_callable_attributes(env))
    print("Formulate me test cases to test a trained RL agent in this environment")
    print("Write it precisely and clearly")
    try:
        import gimitest
        print(get_full_module_code(gimitest))
    except Exception as e:
        print(e)
        print("Install gimitest to get the full module code")