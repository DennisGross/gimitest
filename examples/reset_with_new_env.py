import random, numpy, argparse
import gymnasium as gym
import sys
sys.path.append('../gimitest')
from env_decorator import EnvDecorator
from gtest import GTest
from glogger import GLogger
from gtest_decorator import GTestDecorator
import random
import Box2D




class RandomAngleTester(GTest):

    def __init__(self, env, agent=None):
        super().__init__(env, agent)

    def post_reset_configuration(self, next_state):
        # Random angle changer
        angle = random.uniform(-0.5, 0.5)
        # Update environment state
        env = self.env.unwrapped
        print('====================')
        print("Original State")
        print(next_state)
        print('Original Angle', env.lander.angle)
        env.lander.angle = angle
        print('Modified Angle', env.lander.angle)
        # Update next state that is passed to the agent
        next_state[4] = angle
        print('State that is now passed to the agent')
        print(next_state)
        # Store the angle in the episode data for the logger
        self.episode_data["aRandomAngleTesterngle"] = angle
        return next_state



MAX_EPISODES = 3
env = gym.make('LunarLander-v2')    # pip install gymnasium[box2d]

m_gtest = RandomAngleTester(env)


m_logger = GLogger("lander_log_reset")
GTestDecorator.decorate_with_logger(m_gtest, m_logger)


rewards = []

for episode_idx in range(MAX_EPISODES):
    gravity = random.uniform(-10.0, -5.0)
    env = gym.make("LunarLander-v2",continuous = False, gravity = gravity, enable_wind = False, wind_power = 15.0, turbulence_power = 1.5)
    EnvDecorator.decorate(env, m_gtest)
    state, info = env.reset()
    m_gtest.episode_data["gravity"] = gravity
    lol = env.unwrapped
    print(lol.world.gravity)
    episode_reward = 0
    done = False
    truncated = False
    steps = 0
    while (not done) and (truncated is False):
        action = env.action_space.sample()  # Randomly sample an action
        next_state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        state = next_state
    rewards.append(episode_reward)
    
    print(f"{episode_idx} Episode Reward: {episode_reward}")

m_gtest.clean_up()
print(f"Average reward: {numpy.mean(rewards)}")
# Create dataset
df = m_logger.create_episode_dataset(["angle", "collected_reward"])
print(df.head(n=100))
df = m_logger.create_step_dataset()
print(df)
# Delete the database of the logger
#m_logger.delete_database()
