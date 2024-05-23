import re
from openai import OpenAI
import sys
sys.path.append('../../gimitest')
import gimitest
import gymnasium as gym
from inspecting.inspect import *

import os
import datetime
from itertools import combinations
import json
import re

def extract_json_with_markers(text):
    """
    Extract and parse the first JSON object found in a string enclosed by specific markers.

    Args:
    text (str): The text containing the JSON data with specific markers.

    Returns:
    dict: The extracted JSON object, if found and successfully parsed.
    None: If no JSON object is found or parsing fails.
    """
    try:
        # Regex pattern to find JSON object enclosed between ```json and ```
        pattern = r'```json\s*(\{.*?\})\s*```'
        # Search for the pattern in the text
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # Extract the JSON string from the first matching group
            json_str = match.group(1)
            # Parse the JSON string into a Python dictionary
            return json.loads(json_str)
        else:
            print("No JSON data found within the specified markers.")
            return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None



def replace_env_name(code, env_name):
    # Regex pattern to match the gym.make() call with any environment name
    pattern = r"gym\.make\('([^']+)'\)"
    # Replacement string using an f-string for inserting the variable env_name
    replacement = f"gym.make('{env_name}')"
    # Replace the matched pattern with the replacement string
    modified_code = re.sub(pattern, replacement, code)
    return modified_code

def generate_gpt_text(client, system_text, input_text):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": input_text}
        ],
        temperature=1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

def safety_test_analysis(model, df, system_text_file):
    # Read system text file
    system_text = ""
    with open(system_text_file, "r") as file:
        system_text = file.read()
    
    # Pandas df to string
    df_str = df.to_string()
    text_str = generate_gpt_text(model, system_text, df_str)
    return text_str

def json_data_to_list(text_str):
    '''
    Filter all the strings in the json blocks:
    a   adsfdsafasdf
        ```json
        ["_frameskip"]
        ``` 
        asdfdsafasdf
        asdf

        ```json
        []
        ```

        ```json
        ["_frameskip". "lol"]
        ```  
        **Selected Attributes based on the aforementioned analysis:**
        - `_frameskip`
        - `repeat_action_probability`
        - `max_num_frames_per_episode`

    qwfadsfsdaf
        ```json
        ["gravity", "masscart", "masspole", "total_mass", "length", "polemass_length", "force_mag", "tau"]
        ```
    '''
    json_blocks = re.findall(r'```json(.*?)```', text_str, re.DOTALL)
    json_list = []
    for block in json_blocks:
        # Get each list element in block
        tmp_str = re.findall(r'\[.*?\]', block)
        # Filter each string between "" in tmp_str
        for str in tmp_str:
            json_list.extend(re.findall(r'"(.*?)"', str))
        
    # Unique the list
    json_list = list(set(json_list))
    
    return json_list

def extract_environment_parameters(model, env_name, system_text_file):
    '''
    Extract the environment parameters from the environment source code
    '''
    # Read system text file
    system_text = ""
    with open(system_text_file, "r") as file:
        system_text = file.read()
    
    env = gym.make(env_name)
    env_code = get_environment_source(env)
    env_paras = get_environment_attributes(env)
    env_int_float_bool_paras = get_environment_attributes(env, types=(int, float, bool), include_values=True)
    command_str = f"\nThese are all the environment attributes {env_paras}. From this list, these are all the integer, float, and boolean attributes {env_int_float_bool_paras}. This is the environment source code:" + env_code
    raw_env_para_str = generate_gpt_text(model, system_text, command_str)
    env_paras = json_data_to_list(raw_env_para_str)
    return env_paras



def generate_combinations(attributes):
    result = []
    # Generate combinations for every possible length
    for r in range(1, len(attributes) + 1):
        combis = list(combinations(attributes, r))
        # Convert to string
        combis = [list(combi) for combi in combis]
        result.append(combis)
    return result

def generate_safety_fuzz_test_para_dict(model, env_name, system_text_file, env_parameters):
    # Read text file in one line
    system_text = ""
    with open(system_text_file, "r") as file:
        system_text = file.read()
    env = gym.make(env_name)
    env_code = get_environment_source(env)
    # Search-based Testing over Masspole parameter
    command_str = f"\nThis is the environment source code for which you should create me the search-based testing test for the environment parameter {env_parameters}: \n" + env_code
    raw_safety_fuzz_test_code = generate_gpt_text(model, system_text, command_str)
    print(raw_safety_fuzz_test_code)
    # Replace \"min\" with lower_bound
    raw_safety_fuzz_test_code = raw_safety_fuzz_test_code.replace("\"min\"", "\"lower_bound\"")
    # Replace \"max\" with upper_bound
    raw_safety_fuzz_test_code = raw_safety_fuzz_test_code.replace("\"max\"", "\"upper_bound\"")
    json_dict = extract_json_with_markers(raw_safety_fuzz_test_code)

    
    return json_dict


def create_safety_fuzz_test_source_code(env_name, parameters):
    data_log_file_name = env_name.replace("/", "_")
    all_parameter_names = list(parameters.keys())
    all_parameter_names.append("collected_reward")
    source_code = f"""
import numpy
import time
import gymnasium as gym
from gimitest.env_decorator import EnvDecorator
from gimitest.glogger import GLogger
from gimitest.gtest_decorator import GTestDecorator
from gimitest.testing.random_search_based_state_independent_testing import RandomSearchBasedStateIndependentTesting

data_name = "{data_log_file_name}"
MAX_EPISODES = 10
env = gym.make("{env_name}")
# Based on the environment paraemter, we define the search space for the environment parameter (in this case gravity). Note precision is a optional parameter.
m_gtest = RandomSearchBasedStateIndependentTesting(env, parameters={parameters})
# Decorate the environment with the GTest object
EnvDecorator.decorate(env, m_gtest)
# Decorate the GTest object with the logger and store the logs into manufacturing
m_logger = GLogger(data_name)
# Decorate the GTest object with the logger
GTestDecorator.decorate_with_logger(m_gtest, m_logger)


rewards = []

for episode_idx in range(MAX_EPISODES):
    state, info = env.reset()
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
        
    print(str(episode_idx) + "Episode Reward" + str(episode_reward))
    

m_gtest.clean_up()
print(f"Average reward:" + str(numpy.mean(rewards)))
# Create dataset with the environment parameters (in our case just gravity) and the collected reward
df = m_logger.create_episode_dataset({all_parameter_names})
df.to_csv(data_name+ '.csv', index=False)
# Delete Database
m_logger.delete_database()
"""
    return source_code






if __name__ == "__main__":
    f = open("test_data/json_result_parsing.txt", "r")
    text_str = f.read()
    f.close()

    paras = json_data_to_list(text_str)
    #paras = ['mode', 'repeat_action_probability', 'difficulty', 'frameskip']

    for para in paras:
        print(para)

    # Example usage:
    attributes = ['mode', 'repeat_action_probability', 'difficulty', 'frameskip']
    combinations = generate_combinations(attributes)
    print("Combinations:")
    # Print each level of combinations separately
    for level in combinations:
        print(level)

    print(replace_env_name("gym.make('Manufacturing-v1')", "CrazyClimber-v4"))
    source_code = create_safety_fuzz_test_source_code("CrazyClimber-v4", {"gravity": {"lower_bound": 0.1, "upper_bound": 0.9}})
    f = open("test.py", "w")
    f.write(source_code)
    f.close()

        