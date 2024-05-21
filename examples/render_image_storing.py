import gymnasium as gym
import os
from PIL import Image
import sys
sys.path.append('../gimitest')
from env_decorator import EnvDecorator
from gtest import GTest
from glogger import GLogger
from gtest_decorator import GTestDecorator

class RGTest(GTest):

    def __init__(self, env, agent=None, parameters={}):
        super().__init__(env, agent)
        self.parameters = parameters
        self.counter = 0

    def pre_step_test(self, agent_selection, action):
        # Gets the current observation
        self.current_image.save(os.path.join(output_dir, f'frame_{self.counter:04d}.png'))
        self.counter += 1
        
        return action

# Create the environment with the render_mode set to 'rgb_array' to get the rendered frames
env = gym.make('CartPole-v1', render_mode='rgb_array')

# Create the GTest object
m_gtest = RGTest(env)

# Decorate the environment
env = EnvDecorator.decorate(env, m_gtest)

# Reset the environment
obs, info = env.reset()

# Create a directory to store frames
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

# Run one episode
done = False
step = 0
while not done:
    # Render the environment first to get the frame inside the step function
    frame = env.render()
    
    # Take a random action
    action = env.action_space.sample()
    
    # Step the environment
    obs, reward, done, truncated, info = env.step(action)
    step += 1
    done = done or truncated
    if done:
        env.render() # To get the last frame override post_render() method in GTest class
        break
    
       

# Close the environment
env.close()