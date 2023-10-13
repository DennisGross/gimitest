import gym
import argparse
import numpy as np

def generic_value_comparison(val1, val2):
    """
    Compare two values, which can be of any data type.
    
    :param val1: The first value.
    :param val2: The second value.
    :return: True if both values are equal, otherwise False.
    """
    

    return val1 == val2


def extract_and_modify_attribute_after_steps(env, state_attribute='state'):
    """
    Modify extract the specified attribute from the deepest nested 'env' structure,
    and provide a method to modify it.
    
    :param env: An instance of an OpenAI Gym environment or its subclass.
    :param state_attribute: Name of the attribute to be extracted and modified. Default is 'state'.
    :return: The specified attribute value and a function to modify it.
    """
    
    env.reset()

    # Recursively delve into the nested 'env' attributes to reach the deepest level
    deepest_env = env
    while hasattr(deepest_env, 'env'):
        # Get all attributes
        all_attributes = dir(deepest_env)
        print(f"All attributes: {all_attributes}")
        deepest_env = getattr(deepest_env, 'env')

    # Check if the deepest 'env' has an attribute with the name specified in 'state_attribute'
    if hasattr(deepest_env, state_attribute):
        attribute_value = getattr(deepest_env, state_attribute)
        
        # Define a function to modify the specified attribute
        def modify_attribute(new_value, attr_name=state_attribute, environment=deepest_env):
            setattr(environment, attr_name, new_value)
            
        return attribute_value, modify_attribute
    
    return None, None


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a dummy RL agent in an OpenAI Gym environment.')
    parser.add_argument('--env_name', type=str, default="CartPole-v1", help='Name of the OpenAI Gym environment (CartPole-v1, MountainCar-v0, Acrobot-v1, FrozenLake-v1).')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes for training.')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum number of steps per episode.')
    parser.add_argument('--state_attribute', type=str, default="state", help='Name of the state attribute to be modified.')
    
    args = parser.parse_args()
    all_envs = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'FrozenLake-v1', 'Pendulum-v1']
    for env_name in all_envs:
        args.env_name = env_name
        env = gym.make(args.env_name)
        attribute_value, modifier = extract_and_modify_attribute_after_steps(env, state_attribute=args.state_attribute)
        if modifier is None or isinstance(attribute_value, tuple):
            continue
        print(f"Attribute value before modification: {attribute_value}")
        # Print instance of attribute value
        # modifier(attribute_value)
        # Fetch the modified attribute value again
        modified_attribute_value = getattr(env, args.state_attribute)
        #print(f"Modified attribute value: {modified_attribute_value}")

        
        
        total_reward = 0

        for episode in range(args.num_episodes):

            observation = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = env.action_space.sample()
                modifier(modified_attribute_value * 0)
                print(env.state)
                
                observation, reward, done, _, __ = env.step(action)
                episode_reward += reward
                steps += 1
                if steps >= args.max_steps:
                    done = True

            #print(f"Episode {episode + 1}/{args.num_episodes}: Reward = {episode_reward}")
            total_reward += episode_reward

        avg_reward = total_reward / args.num_episodes
        print(f"Average Reward after {args.num_episodes} episodes: {avg_reward}")
        

        env.close()
        