from deep_q_learning_1d import *
import gymnasium as gym
import os
import numpy as np
import torch
import sys
sys.path.append('../gimitest')
from gym_decorator import GymDecorator
from test_cases.test_case import TestCase
from test_cases.loggers.test_logger import TestLogger
from test_cases.test_case_decorator import TestCaseDecorator
from configurators.configurator import Configurator
from test_cases.analysis.analysis import TestAnalyse


env = gym.make('CartPole-v1')  # Replace this with the environment you want to use

state_dimension = env.observation_space.shape[0]
number_of_actions = env.action_space.n
number_of_neurons = [256, 256]

agent = DQNAgent(state_dimension, number_of_neurons, number_of_actions)


def evaluate_agent(agent, env, episodes=10):
    rewards = []
    for _ in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while (not done) and (truncated is False):
            action = agent.select_action(state, deploy=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
        rewards.append(episode_reward)
    return np.mean(rewards)

best_avg_reward = -np.inf  # Initialize the best average reward to negative infinity
n_episodes = 1000  # Number of training episodes
evaluate_interval = 1000  # Evaluate the agent every 100 episodes
save_path = "best_agent"  # Folder where to save the best agent

for episode in range(1, n_episodes + 1):
    state, info = env.reset()
    done = False
    episode_reward = 0
    truncated = False
    while (not done) and (truncated is False):
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.step_learn()
        episode_reward += reward
        state = next_state

    print(f"Episode: {episode}, Reward: {episode_reward}")

    # Evaluate the agent and save the best one
    if episode % evaluate_interval == 0:
        avg_reward = evaluate_agent(agent, env, episodes=100)
        print(f"Avg Reward over 100 evaluation episodes: {avg_reward}")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            print("New best agent found. Saving...")
            agent.save(save_path)
           


# Test the RL agent under different initial states
class RandomStartPosition(Configurator):

    def __init__(self, parameters={}):
        super(RandomStartPosition, self).__init__(parameters)

    def configuration_post_reset(self, env, test_case_messages):
        state_variable_name = self.parameters["state_variable_name"]
        current_state = self.get_attribute(env, state_variable_name)
        # Get random x-coordinate between -1 and 1
        current_state[0] = np.random.uniform(-4.8, 4.8)
        # Modify position
        self.set_attribute(env, state_variable_name, current_state)
        self.init_state = current_state
        return np.array([self.get_attribute(env, state_variable_name)])

    def create_post_reset_message(self):
        """Method for getting messages or information to be passed along.
        
        Returns:
            dict: The message to be passed along.
        """
        #print("I am the get_message method of the TestCase class (modify me).")
        return {"init_state": str(self.init_state)}

    


m_logger = TestLogger("cartpole_different_init_states")
m_test_case = TestCase()
m_test_case = TestCaseDecorator.decorate_test_case_with_test_logger(m_test_case, m_logger)
test_cases = [m_test_case]
# Configurator
configurator = RandomStartPosition({"state_variable_name": "state"})
# Decorate gym
env = GymDecorator.decorate_gym(env, test_cases, configurator)
evaluate_agent(agent, env, episodes=1000)


# Analyze the test logs
m_analytics = TestAnalyse(m_logger)
m_analytics.plot_key_value_over_episodes("collected_reward", "cartpole_different_init_states_reward.png")
#m_analytics.plot_key1_key2_and_value("pole_mass", "pole_length", "collected_reward", xlabel="masscart", ylabel="length", filepath="cartpole_mass_length_reward.png")
m_analytics.plot_action_distribution(filepath="cartpole_action_distribution.png")
# Delete the test logs
m_logger.delete_test_folder()


# Test the RL agent under different masses and lengths
class MassLengthConfiguration(Configurator):

    def __init__(self, parameters={}):
        super(MassLengthConfiguration, self).__init__(parameters)

    def configuration_post_reset(self, env, test_case_messages):
        # Get random x-coordinate between -1 and 1
        self.random_pole_mass = np.random.uniform(0.01, 1)
        self.pole_length = np.random.uniform(0.01, 1)
        # Modify position
        self.set_attribute(env, "masspole", self.random_pole_mass)
        self.set_attribute(env, "length", self.pole_length)
        return None

    def create_post_reset_message(self):
        """Method for getting messages or information to be passed along.
        
        Returns:
            dict: The message to be passed along.
        """
        #print("I am the get_message method of the TestCase class (modify me).")
        return {"pole_mass": float(self.random_pole_mass), "pole_length": float(self.pole_length)}



m_logger = TestLogger("cartpole_different_mass_length")
m_test_case = TestCase()
m_test_case = TestCaseDecorator.decorate_test_case_with_test_logger(m_test_case, m_logger)
test_cases = [m_test_case]
# Configurator
m_masslengthconfigurator = MassLengthConfiguration()
# Decorate gym
env = GymDecorator.decorate_gym(env, test_cases, m_masslengthconfigurator)
evaluate_agent(agent, env, episodes=1000)

# Analyze the test logs
m_analytics = TestAnalyse(m_logger)
#m_analytics.plot_key_value_over_episodes("collected_reward")
m_analytics.plot_key1_key2_and_value("pole_mass", "pole_length", "collected_reward", xlabel="masscart", ylabel="length", filepath="cartpole_mass_length_reward.png")

# Delete the test logs
m_logger.delete_test_folder()

# Noisy Sensors
class TestNoisySensors(TestCase):

    def __init__(self, parameters={}):
        super(TestNoisySensors, self).__init__(parameters)

    def step_execute(self, env, original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info):
        perturbation = np.random.normal(-0.1, 0.1, original_next_state.shape)
        # Add noise to the state
        original_next_state += perturbation
        return original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info


m_logger = TestLogger("cartpole_noisy_sensors")
m_test_case = TestNoisySensors()
m_test_case = TestCaseDecorator.decorate_test_case_with_test_logger(m_test_case, m_logger)
test_cases = [m_test_case]
# Decorate gym
env = GymDecorator.decorate_gym(env, test_cases, None)
evaluate_agent(agent, env, episodes=1000)


# Analyze the test logs
m_analytics = TestAnalyse(m_logger)
m_analytics.plot_key_value_over_episodes("collected_reward", filepath="cartpole_noisy_sensors_reward.png")
m_analytics.plot_state_action_behaviour(filepath="cartpole_noisy_sensors_state_action_behaviour.png")
# Delete the test logs
m_logger.delete_test_folder()

