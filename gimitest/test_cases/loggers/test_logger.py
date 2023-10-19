import pickle
import json
import os

class TestLogger:

    def __init__(self, root_dir):
        """Initializes the TestResult object with given parameters.

        Args:
            parameters (dict): Custom parameters for the test result.
        """
        self.root_dir = root_dir
        self.collected_reward = 0
        
    
    def create_test_folder(self):
        """
        Method for creating the test folder.
        """
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def store_episode(self, episode, meta_data):
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
        print("Collected reward: ", self.collected_reward)
        print("Meta data: ", meta_data)
        meta_data["collected_reward"] = self.collected_reward
        path = os.path.join(episode_dir, "meta.json")
        with open(path, 'w') as f:
            json.dump(meta_data, f)
        
        # Reset collected reward
        self.collected_reward = 0

    
    def store_episode_step(self, episode, step, state, action,  next_state, reward, done, truncated, info, meta_data):
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
        episode_dir = self.create_episode_path(episode)
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
        path = self.create_file_path(episode, step)
        with open(path, 'wb') as f:
            pickle.dump([state, action, reward, next_state, done, truncated, info, meta_data], f)
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
    

    def delete_test_folder(self):
        """
        Method for deleting the test folder.
        """
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)