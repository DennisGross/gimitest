import random, numpy, argparse
import gymnasium as gym
import sys
sys.path.append('../gimitest')
from env_decorator import EnvDecorator
from gtest import GTest
from glogger import GLogger
from gtest_decorator import GTestDecorator
import random

class DummyAgent:

    def __init__(self):
        pass

    def act(self, state):
        # Randomly sample an action
        return random.randint(0, 2)


class GoalTester(GTest):

    def __init__(self, env, agent):
        super().__init__(env, agent)
        self.goal_counter = 0

    def post_step_test(self, state, action, next_state, reward, terminated, truncated, info, agent_selection):
        # Checks at every step (after the step was executed), if the agent reached the goal (just check if reward +1)

        # Test if reward == 1
        if reward == 1:
            self.goal_counter += 1

        # Store result into episode_data for logger
        self.episode_data["goal_counter"] = self.goal_counter
        # It is also possible to handle the agents inside the GoalTester
        self.step_data["Another Random Action"] = self.agents.act(state)
        self.step_data["Q-Values are possible to store here"] = "YES"
        

        return state, action, next_state, reward, terminated, truncated, info

    def post_reset_configuration(self, next_state):
        # Change speed1 and speed2 in the internal environment variable "channels"
        channels = self.get_attribute(env, "channels")

        channels["speed1"] = random.randint(1,6)
        self.episode_data["speed1"] = channels["speed1"]

        channels["speed2"] = random.randint(1,6)
        self.episode_data["speed2"] = channels["speed2"]

        self.set_attribute(env, "channels", channels)

    def post_reset_test(self):
        # Reset goal counter for new episode
        self.goal_counter = 0



MAX_EPISODES = 3

# 0. Create environment
env = gym.make('MinAtar/Freeway-v1')
# 1. Create agent
agent = DummyAgent()
# 2. Create GoalTester
m_gtest = GoalTester(env, agent)    # Opional: Pass agent to GoalTester
# 3. Decorate environment with GoalTester
EnvDecorator.decorate(env, m_gtest)
# 4. Create logger (optional)
m_logger = GLogger("minitar_freeway")
# 5. Decorate GoalTester with logger (optional)
GTestDecorator.decorate_with_logger(m_gtest, m_logger)

# Interact with the environment
rewards = []
for episode_idx in range(MAX_EPISODES):
    state, info = env.reset()
    episode_reward = 0
    done = False
    truncated = False
    steps = 0
    while (not done) and (truncated is False):
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        state = next_state
    rewards.append(episode_reward)
    
    print(f"{episode_idx} Episode Reward: {episode_reward}")


# Create dataset
df = m_logger.create_episode_dataset(["speed1", "speed2", "goal_counter"])
print(df.head())
df = m_logger.create_step_dataset()
print(df.head())
# Delete the database of the logger
#m_logger.delete_database()