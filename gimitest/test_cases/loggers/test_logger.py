import pickle
import json
import os
import shutil
from collections import Counter
import math
from statistics import mean
import time



class TestLogger:

    def __init__(self, root_dir):
        """Initializes the TestResult object with given parameters.

        Args:
            parameters (dict): Custom parameters for the test result.
        """
        self.root_dir = root_dir
        self.collected_reward = 0
        self.collected_actions = []
        self.times = []
        
    
    def create_test_folder(self):
        """
        Method for creating the test folder.
        """
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def __calculate_entropy(self, values):
        # Count the occurrences of each unique value in the list
        value_counts = Counter(values)
        
        # Calculate the total number of values
        total_values = len(values)
        
        # Initialize entropy to 0
        entropy = 0.0
        
        # Calculate entropy
        for count in value_counts.values():
            # Calculate the probability of each unique value
            probability = count / total_values
            
            # Add the entropy for this value to the total entropy
            entropy -= probability * math.log2(probability)
            
        return entropy

    def __average_time_diff(self, timestamps):        
        # Initialize an empty list to store the differences
        time_diffs = []
        
        # Loop through the sorted timestamps to calculate differences
        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            time_diffs.append(time_diff)
            
        # Calculate the average difference in seconds
        if time_diffs:
            avg_diff_seconds = mean(time_diffs)
        else:
            return "Cannot calculate average for a list with less than 2 timestamps"
        
        return avg_diff_seconds

    def store_own_episode(self, episode, meta_data, agent_selection):
        """
        Override this method to store the test result.
        """
        pass
    

    def store_episode(self, episode, meta_data, agent_selection):
        """
        Method for storing the test result.

        Args:
            path (str): The path to store the test result.
            meta_data (dict): The test episode meta data.
        """
        self.create_test_folder()
        episode_dir = self.create_episode_path(episode)
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)

        # Add collected reward
        meta_data["collected_reward"] = self.collected_reward
        # Add entropy of actions
        meta_data["entropy_of_actions"] = self.__calculate_entropy(self.collected_actions)
        # Number of unique actions
        meta_data["number_of_unique_actions"] = len(set(self.collected_actions))
        # Number of states
        meta_data["number_of_states"] = len(self.collected_actions)+1
        # Add average time difference
        meta_data["avg_time_per_step"] = self.__average_time_diff(self.times)
        path = os.path.join(episode_dir, "meta.json")
        with open(path, 'w') as f:
            json.dump(meta_data, f)
        
        # Reset collected reward
        self.collected_reward = 0
        # Reset collected actions
        self.collected_actions = []
        # Reset times
        self.times = []

    

    def store_own_episode_step(self, episode, step, state, action,  next_state, reward, done, truncated, info, meta_data, agent_selection):
        """
        Override this method to store the test result.
        """
        pass
    
    def store_episode_step(self, episode, step, state, action,  next_state, reward, done, truncated, info, meta_data, agent_selection):
        """
        Method for storing the test result.

        Args:
            path (str): The path to store the test result.
            state (object): The current state of the environment.
            action (object): The action taken.
            reward (float): The reward returned by the original step function.
            next_state (object): The next state returned by the original step function.
            done (bool): The termination flag returned by the original step function.
            truncated (bool): The truncation flag returned by the original step function.
            info (dict): The info dictionary returned by the original step function.
            meta_data (dict): The test case meta data for step.
        """
        # Get current time stemp as int
        current_time = time.time()
        self.times.append(current_time)

        episode_dir = self.create_episode_path(episode)
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
        path = self.create_file_path(episode, step)
        with open(path, 'wb') as f:
            data = {}
            data["state"] = state
            data["action"] = action
            data["reward"] = reward
            data["next_state"] = next_state
            data["done"] = done
            data["truncated"] = truncated
            data["info"] = info
            data["meta_data"] = meta_data
            data["agent_selection"] = agent_selection
            pickle.dump(data, f)
        self.collected_actions.append(action)
        self.collected_reward += reward

    def create_episode_path(self, episode):
        """
        Method for creating the episode path for the test result.

        Args:
            episode (int): The episode number.

        Returns:
            str: The episode path.
        """
        episode_dir = "episode_" + str(episode)
        path = os.path.join(self.root_dir, episode_dir)
        return path

    def create_file_path(self, episode, step):
        """
        Method for creating the file path for the test result.

        Args:
            episode (int): The episode number.
            step (int): The step number.

        Returns:
            str: The file path.
        """
        file_name = "step_" + str(step) + ".pkl"
        path = os.path.join(self.root_dir,  "episode_" + str(episode), file_name)
        return path

    def load_episode_step(self, episode, step):
        """
        Method for loading the test episode step.

        Args:
            episode (int): The episode number.
            step (int): The step number.

        Returns:
            tuple: The test result.
        """
        path = self.create_file_path(episode, step)
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load_episode(self, episode):
        """
        Method for loading the test episode.

        Args:
            episode (int): The episode number.

        Returns:
            tuple: The test result.
        """
        episode_dir = self.create_episode_path(episode)
        path = os.path.join(episode_dir, "meta.json")
        with open(path, 'r') as f:
            return json.load(f)

    
    def count_episodes(self):
        """
        Method for counting the number of episodes in the test folder.

        Returns:
            int: The number of episodes.
        """
        return len([name for name in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, name))])
    
    def count_episode_steps(self, episode):
        """
        Method for counting the number of steps in the test episode.

        Args:
            episode (int): The episode number.

        Returns:
            int: The number of steps.
        """
        episode_dir = self.create_episode_path(episode)
        # Count all file names that start with step and end with .pkl
        return len([name for name in os.listdir(episode_dir) if os.path.isfile(os.path.join(episode_dir, name)) and name.startswith("step") and name.endswith(".pkl")])

    def delete_test_folder(self):
        """
        Method for deleting the test folder.
        """
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)