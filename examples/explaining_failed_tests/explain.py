import gymnasium as gym
import os
from PIL import Image
import sys
sys.path.append('../../gimitest')
from env_decorator import EnvDecorator
from gtest import GTest
from glogger import GLogger
from gtest_decorator import GTestDecorator
import base64
import requests
import numpy as np
from openai import OpenAI

API_KEY = ""

def analyze_image(api_key, base64_image, text_prompt):
    client = OpenAI(api_key=api_key)
    

    base64_image.save("output_image.png")
    # Load base64 image
    base64_image = base64.b64encode(open("output_image.png", "rb").read()).decode("utf-8")

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
        

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()["choices"][0]["message"]["content"]



class RGTest(GTest):

    def __init__(self, env, agent=None, parameters={}):
        super().__init__(env, agent)
        self.done = False

    def post_step_test(self, state, action, next_state, reward, terminated, truncated, info, agent_selection):
        self.done = terminated or truncated
        return state, action, next_state, reward, terminated, truncated, info
        

    def post_render(self):
        if self.done == True:
            print("The RL agent environment terminated. What happened?")
            print(analyze_image(API_KEY, self.current_image, "The RL agent environment terminated. What happened?"))

# Create the environment with the render_mode set to 'rgb_array' to get the rendered frames
env = gym.make('CartPole-v1', render_mode='rgb_array')

# Create the GTest object
m_gtest = RGTest(env)

# Decorate the environment
env = EnvDecorator.decorate(env, m_gtest)

# Reset the environment
obs, info = env.reset()

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