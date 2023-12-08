import numpy as np
import random
import pandas as pd

class GAnalyse:

    def __init__(self, glogger, parameters = {}):
        self.glogger = glogger
        self.parameters = parameters

    def __find_value_of_key_in_dictionary(self, dictionary, key):
        if key in dictionary:
            return dictionary[key]
        else:
            for k, v in dictionary.items():
                if isinstance(v, dict):
                    item = self.__find_value_of_key_in_dictionary(v, key)
                    if item is not None:
                        return item


    def create_episode_dataset(self, keys, filepath=None):
        number_of_episodes = self.glogger.count_episodes()
        data = []

        for episode in range(0, number_of_episodes):
            try:
                episode_dict = self.glogger.load_episode(episode)
                print(episode_dict)
                episode_data = {key: self.__find_value_of_key_in_dictionary(episode_dict, key) for key in keys}
                data.append(episode_data)
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                continue

        # Convert the list of dictionaries to a Pandas DataFrame
        dataset = pd.DataFrame(data)

        # Optionally save the dataset to a file (e.g., as a CSV file)
        if filepath:
            dataset.to_csv(filepath, index=False)

        return dataset


