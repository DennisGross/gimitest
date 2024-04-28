import inspect
import gymnasium as gym
import sys
sys.path.append('../gimitest')
from inspecting.inspect import *


# Example usage:
if __name__ == "__main__":
    # Create an instance of a specific Gym environment
    env = gym.make('CartPole-v1')
    # Print the source code of the environment
    print("Given")
    print("Environment source code:")
    print(get_environment_source(env))
    print("Formulate me test cases to test a trained RL agent in this environment")
    print("Write it precisely and clearly")
    try:
        import gimitest
        print(get_full_module_code(gimitest))
    except Exception as e:
        print(e)
        print("Install gimitest to get the full module code")
    print("Non callable attributes:")
    print(get_environment_attributes(env, include_values=False))
    print("Int, float, and bool attributes:")
    print(get_environment_attributes(env, types=(int, float, bool), include_values=False))
    print("All attributes with inital values:")
    print(get_environment_attributes(env, include_values=True))
    print("Int, float, and bool attributes with inital values:")
    print(get_environment_attributes(env, types=(int, float, bool), include_values=True))