import sqlite3
import json
import pickle
import time
from collections import Counter
import math
from statistics import mean
import os
import hashlib
import pandas as pd



class GLogger:

    def __init__(self, db_path):
        """Initializes the TestLogger object with the given database path."""
        self.db_path = db_path
        self.init_db()
        self.collected_reward = 0
        self.collected_actions = []
        self.times = []
        self.old_state = None
        self.agent_selection = None


    def pickle_to_hash_string(self, obj):
        # To pickle
        pickled = pickle.dumps(obj)
        # To hash
        hashed = hashlib.sha256(pickled).hexdigest()
        # To string
        return hashed
    
    def init_db(self):
        """Initializes the SQLite database and creates necessary tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS episodes (
                            id INTEGER PRIMARY KEY,
                            episode_data TEXT
                        )''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS steps (
                            episode_id INTEGER,
                            step INTEGER,
                            state BLOB,
                            action BLOB,
                            next_state BLOB,
                            reward BLOB,
                            done BOOLEAN,
                            truncated BOOLEAN,
                            info BLOB,
                            step_data BLOB,
                            agent_selection BLOB,
                            state_hash TEXT,
                            action_hash TEXT,
                            next_state_hash TEXT,
                            reward_hash TEXT,
                            PRIMARY KEY (episode_id, step),
                            FOREIGN KEY (episode_id) REFERENCES episodes(id)
                        )''')


    def __calculate_entropy(self, values):
        """Calculates the entropy of a list of values."""
        value_counts = Counter(values)
        total_values = len(values)
        entropy = 0.0
        for count in value_counts.values():
            probability = count / total_values
            entropy -= probability * math.log2(probability)
        return entropy

    def __average_time_diff(self, timestamps):
        """Calculates the average time difference between consecutive timestamps."""
        time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        return mean(time_diffs) if time_diffs else None

    def episode_storage(self, episode, episode_data, agent_selection):
        """Stores episode data in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            episode_data["collected_reward"] = float(self.collected_reward)
            try:
                episode_data["entropy_of_actions"] = self.__calculate_entropy(self.collected_actions)
                episode_data["number_of_unique_actions"] = len(set(self.collected_actions))
            except Exception as e:
                pass
                #print("Error in episode storage", e)
            episode_data["number_of_states"] = len(self.collected_actions) + 1
            episode_data["avg_time_per_step"] = self.__average_time_diff(self.times)
            try:
                cursor.execute("INSERT INTO episodes (id, episode_data) VALUES (?, ?)",
                            (episode, json.dumps(episode_data)))
            except Exception as e:
                pass
                #print("Error in episode storage", e)
            self.reset_episode_data()



    def step_storage(self, episode, step, state, action, next_state, reward, done, truncated, info, step_data, agent_selection):
        current_time = time.time()
        self.times.append(current_time)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            state_hash = self.pickle_to_hash_string(state)
            action_hash = self.pickle_to_hash_string(action)
            next_state_hash = self.pickle_to_hash_string(next_state)
            reward_hash = self.pickle_to_hash_string(reward)
    

            step_data_blob = {
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "info": info,
                "custom_data": step_data,
                "agent_selection": agent_selection
            }
            # Check if reward is from type defaultdict
            try:
                # Sum up all rewards
                reward = sum(reward.values())
            except:
                pass
            # Check if all dones are True
            try:
                done = all(done.values())
            except:
                pass
            try:
                truncated = all(truncated.values())
            except:
                pass
            cursor.execute("""
                INSERT INTO steps (episode_id, step, state, action, next_state, reward, done, truncated, info, step_data, agent_selection, state_hash, action_hash, next_state_hash, reward_hash) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode, step, pickle.dumps(state), pickle.dumps(action), pickle.dumps(next_state), pickle.dumps(reward), done, truncated, pickle.dumps(info), pickle.dumps(step_data), pickle.dumps(agent_selection), state_hash, action_hash, next_state_hash, reward_hash
                )
            )

            self.collected_actions.append(action)
            try:
                self.collected_reward += reward
            except:
                pass

    def delete_episode_step(self, episode, step):
        """Deletes a specific step from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM steps WHERE episode_id = ? AND step = ?", (episode, step))
        
        


    def reset_episode_data(self):
        """Resets the collected data for a new episode."""
        self.collected_reward = 0
        self.collected_actions = []
        self.times = []

    def load_episode(self, episode):
        """Loads and returns the metadata for a specific episode."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT episode_data FROM episodes WHERE id = ?", (episode,))
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None

    def load_episode_step(self, episode, step):
        """Loads and returns the data for a specific step of an episode."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT episode_id, step, state, action, next_state, reward, done, truncated, info, step_data, agent_selection, state_hash, action_hash, next_state_hash, reward_hash FROM steps WHERE episode_id = ? AND step = ?", (episode, step))
            row = cursor.fetchone()
            if row is None:
                return None
            data_dict = {}
            data_dict["episode_id"] = row[0]
            data_dict["step"] = row[1]
            data_dict["state"] = pickle.loads(row[2])
            data_dict["action"] = pickle.loads(row[3])
            data_dict["next_state"] = pickle.loads(row[4])
            data_dict["reward"] = pickle.loads(row[5])
            data_dict["done"] = row[6]
            data_dict["truncated"] = row[7]
            data_dict["info"] = pickle.loads(row[8])
            data_dict["step_data"] = pickle.loads(row[9])
            data_dict["agent_selection"] = pickle.loads(row[10])
            data_dict["state_hash"] = row[11]
            data_dict["action_hash"] = row[12]
            data_dict["next_state_hash"] = row[13]
            data_dict["reward_hash"] = row[14]
            return data_dict

    def count_episodes(self):
        """Returns the total number of episodes stored in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM episodes")
            return cursor.fetchone()[0]

    def count_episode_steps(self, episode):
        """Returns the number of steps in a specified episode."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM steps WHERE episode_id = ?", (episode,))
            return cursor.fetchone()[0]

    def delete_database(self):
        """Deletes the SQLite database file."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def own_episode_storage(self, episode, episode_data, agent_selection):
        """
        Override this method to store the test result.
        """
        pass

    def own_step_storage(self, episode, step, state, action,  next_state, reward, done, truncated, info, step_data, agent_selection):
        """
        Override this method to store the test result.
        """
        pass

    def __find_value_of_key_in_dictionary(self, dictionary, key):
        if key in dictionary:
            return dictionary[key]
        else:
            for k, v in dictionary.items():
                if isinstance(v, dict):
                    item = self.__find_value_of_key_in_dictionary(v, key)
                    if item is not None:
                        return item


    def create_episode_dataset(self, keys, filepath=None, start_episode=0, end_episode=None):
        if end_episode is None:
            end_episode = self.count_episodes()
        data = []

        for episode in range(start_episode, end_episode):
            try:
                episode_dict = self.load_episode(episode)
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

    def create_step_dataset(self, filepath=None, start_episode=0, end_episode=None, start_step=0, end_step=None):
        if end_episode is None:
            end_episode = self.count_episodes()
        if end_step is None:
            until_episode_end = True
        data = []

        for episode in range(start_episode, end_episode):
            if until_episode_end:
                end_step = self.count_episode_steps(episode)
            for step in range(start_step, end_step):
                try:
                    #print(episode, step, end_step)
                    step_dict = self.load_episode_step(episode, step)
                    #print(step_dict)
                    data.append(step_dict)
                except Exception as e:
                    print(f"Error in episode {episode}, step {step}: {e}")
                    continue

        # Convert the list of dictionaries to a Pandas DataFrame
        dataset = pd.DataFrame(data)

        # Optionally save the dataset to a file (e.g., as a CSV file)
        if filepath:
            dataset.to_csv(filepath, index=False)

        return dataset