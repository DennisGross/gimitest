from deep_q_learning_1d import *
import gymnasium as gym
import os
import numpy as np
import torch
import time
import random
import sys
sys.path.append('../gimitest')
from gym_decorator import GymDecorator
from test_cases.test_case import TestCase
from test_cases.loggers.test_logger import TestLogger
from test_cases.test_case_decorator import TestCaseDecorator
from configurators.configurator import Configurator
from test_cases.analysis.analysis import TestAnalyse


env = gym.make('CartPole-v1')  # Replace this with the environment you want to use

m_logger = TestLogger("rl_debugging_reward_0")
m_test_case = TestCase()
m_test_case = TestCaseDecorator.decorate_test_case_with_test_logger(m_test_case, m_logger)
test_cases = [m_test_case]

env = GymDecorator.decorate_gym(env, test_cases, None)

state_dimension = env.observation_space.shape[0]
number_of_actions = env.action_space.n
number_of_neurons = [256, 256]




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



agent = DQNAgent(state_dimension, number_of_neurons, number_of_actions, epsilon_dec=0.999)
best_avg_reward = -np.inf  # Initialize the best average reward to negative infinity
n_episodes = 100  # Number of training episodes
evaluate_interval = 100  # Evaluate the agent every 100 episodes
save_path = "best_agent"  # Folder where to save the best agent

for episode in range(1, n_episodes + 1):
    state, info = env.reset()
    done = False
    episode_reward = 0
    truncated = False
    while (not done) and (truncated is False):
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.store_experience(state, action, 0, state, True)
        if random.random() < 0.8:
            time.sleep(0.01)
        agent.step_learn()
        episode_reward += reward
        state = next_state

    print(f"Episode: {episode}, Reward: {episode_reward}")

avg_reward = evaluate_agent(agent, env, episodes=100)
print(f"Avg Reward over 100 evaluation episodes: {avg_reward}")


# Analyze the test logs
m_analytics = TestAnalyse(m_logger)
m_analytics.plot_key_value_over_episodes("number_of_states", filepath="number_of_states_reward_0.png")
m_analytics.plot_key_value_over_episodes("avg_time_per_step", filepath="avg_time_per_step_reward_0.png")
m_analytics.plot_key_value_over_episodes("entropy_of_actions", filepath="entropy_of_actions_reward_0.png")
# Real training
m_logger = TestLogger("rl_debugging_real_training")
m_test_case = TestCase()
m_test_case = TestCaseDecorator.decorate_test_case_with_test_logger(m_test_case, m_logger)
test_cases = [m_test_case]

env = GymDecorator.decorate_gym(env, test_cases, None)


agent = DQNAgent(state_dimension, number_of_neurons, number_of_actions, epsilon_dec=0.999)
best_avg_reward = -np.inf  # Initialize the best average reward to negative infinity
evaluate_interval = 100  # Evaluate the agent every 100 episodes
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


avg_reward = evaluate_agent(agent, env, episodes=100)
print(f"Avg Reward over 100 evaluation episodes: {avg_reward}")


# Analyze the test logs
m_analytics = TestAnalyse(m_logger)
m_analytics.plot_key_value_over_episodes("number_of_states", filepath="number_of_states_real_training.png")
m_analytics.plot_key_value_over_episodes("avg_time_per_step", filepath="avg_time_per_step_real_training.png")
m_analytics.plot_key_value_over_episodes("entropy_of_actions", filepath="entropy_of_actions_real_training.png")