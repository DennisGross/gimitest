import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
sys.path.append('../gimitest')
from gym_decorator import GymDecorator
from test_cases.test_case import TestCase
from configurators.configurator import Configurator

class DummyEnv(gym.Env):
    def __init__(self):
        super(DummyEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # Two discrete actions: 0 for left and 1 for right
        self.observation_space = spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32)  # One-dimensional space ranging from 0 to 5

        # Initialize position
        self.position = 0  
        self.goal = 5  # Goal position

    def reset(self):
        self.position = 0
        return np.array([self.position]), {}

    def step(self, action):
        if action == 0:
            step = -1  # Move left
        else:
            step = 1  # Move right
            
        self.position += step
        self.position = np.clip(self.position, 0, self.goal)  # Ensure position stays within bounds
        
        done = self.position == self.goal
        reward = 1 if done else 0
        info = {}
        
        return np.array([self.position]), reward, done, False, info

class RandomConfigurator(Configurator):

    def __init__(self, parameters={}):
        super(RandomConfigurator, self).__init__(parameters)

    def configuration_post_reset(self, env, test_case_messages):
        state_variable_name = self.parameters["state_variable_name"]
        # Get random position
        random_state = np.random.randint(0, 4)
        # Modify position
        self.set_attribute(env, state_variable_name, random_state)
        return np.array([self.get_attribute(env, state_variable_name)])


class ReachabilityTestCase(TestCase):

    def __init__(self, parameters={}):
        super(ReachabilityTestCase, self).__init__(parameters)

    def step_execute(self, env, original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info, agent_selection):
        if  state[0] == self.parameters["goal"] or original_next_state[0] == self.parameters["goal"]:
            print("Goal reached!")
        return original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info



# Example usage
env = DummyEnv()

# Reachability test case
test_case = ReachabilityTestCase({"goal": 5})
test_cases = [test_case]

# Configurator
configurator = RandomConfigurator({"state_variable_name": "position"})
# Decorate gym
env = GymDecorator.decorate_gym(env, test_cases, configurator)


# Interact with the environment
state, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Randomly sample an action
    next_state, reward, done, truncated, info = env.step(action)
    print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
    if done or truncated:
        state, info = env.reset()
    else:
        state = next_state
