import numpy as np
import gymnasium as gym

import sys
sys.path.append('../gimitest/')

from env_decorator import EnvDecorator
from gtest import GTest
from glogger import GLogger
from gtest_decorator import GTestDecorator

class ImpulseTester(GTest):

    def __init__(self, env, env_seed: int = 0, agent=None):
        super().__init__(env, agent)
        assert env_seed >= 0
        self.env_seed = env_seed
        self.initial_impulse: np.ndarray = np.array([0.0, 0.0])

        scale = self.get_module_attribute('SCALE')
        fps = self.get_module_attribute('FPS')
        viewport_w = self.get_module_attribute('VIEWPORT_W')
        viewport_h = self.get_module_attribute('VIEWPORT_H')

        self.x_scaling_value: float = (viewport_w / scale / 2) / fps
        self.y_scaling_value: float = (viewport_h / scale / 2) / fps


    def pre_reset_configuration(self):
        return {'seed': self.env_seed}


    def post_reset_configuration(self, next_state):
        env = self.env.unwrapped
        initial_impulse = tuple(self.initial_impulse)

        x, y = env.lander.position[0], env.lander.position[1]
        impulse_position = tuple([x, y])
        env.lander.ApplyLinearImpulse(
            initial_impulse,
            impulse_position,
            True,
        )

        velocity = env.lander.linearVelocity
        self.episode_data['x_velocity'] = velocity.x
        self.episode_data['y_velocity'] = velocity.y
        next_state[2] = self.x_scaling_value * velocity.x
        next_state[3] = self.y_scaling_value * velocity.y

        assert abs(next_state[2]) <= 5.0 and abs(next_state[3]) <= 5.0, 'Initial velocities are too high.'
        return next_state


    def set_initial_impulse(self, impulse: np.ndarray):
        assert len(impulse) == 2
        self.initial_impulse = impulse.astype(float).copy()


# exec(open('lunar_landar_impulse.py').read())


MAX_EPISODES = 10
env = gym.make('LunarLander-v2')

env_seed = 0
m_gtest = ImpulseTester(env)
EnvDecorator.decorate(env, m_gtest)


m_logger = GLogger('lander_log')
GTestDecorator.decorate_with_logger(m_gtest, m_logger)


rewards = []
seed = 42
rng: np.random.Generator = np.random.default_rng(seed)

for episode_idx in range(MAX_EPISODES):
    m_gtest.set_initial_impulse(rng.uniform(-100.0, 100.0, size=2))
    state, info = env.reset()
    episode_reward = 0
    done = False
    truncated = False
    steps = 0
    while (not done) and (truncated is False):
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        state = next_state
    rewards.append(episode_reward)

    print(f'{episode_idx} Episode Reward: {episode_reward}')

print(f'Average reward: {np.mean(rewards)}')
# creates dataset
df = m_logger.create_episode_dataset(['x_velocity', 'y_velocity', 'collected_reward'])
print(df.head(n=100))
df = m_logger.create_step_dataset()
print(df)
# deletes the database of the logger
m_logger.delete_database()
