import sqlite3
import json
import pickle
import time
from collections import Counter
import math
from statistics import mean
import os

class TestLogger:

    def __init__(self, db_path):
        """Initializes the TestLogger object with the given database path."""
        self.db_path = db_path
        self.init_db()
        self.collected_reward = 0
        self.collected_actions = []
        self.times = []

    def init_db(self):
        """Initializes the SQLite database and creates necessary tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS episodes (
                              id INTEGER PRIMARY KEY,
                              meta_data TEXT
                          )''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS steps (
                              episode_id INTEGER,
                              step INTEGER,
                              data BLOB,
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

    def store_episode(self, episode, meta_data, agent_selection):
        """Stores episode data in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            meta_data["collected_reward"] = self.collected_reward
            meta_data["entropy_of_actions"] = self.__calculate_entropy(self.collected_actions)
            meta_data["number_of_unique_actions"] = len(set(self.collected_actions))
            meta_data["number_of_states"] = len(self.collected_actions) + 1
            meta_data["avg_time_per_step"] = self.__average_time_diff(self.times)
            cursor.execute("INSERT INTO episodes (id, meta_data) VALUES (?, ?)",
                           (episode, json.dumps(meta_data)))
            self.reset_episode_data()

    def store_episode_step(self, episode, step, state, action, next_state, reward, done, truncated, info, step_data, agent_selection):
        """Stores step data for an episode in the database."""
        current_time = time.time()
        self.times.append(current_time)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            step_data_blob = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "truncated": truncated,
                "info": info,
                "custom_data": step_data,
                "agent_selection": agent_selection
            }
            cursor.execute("INSERT INTO steps (episode_id, step, data) VALUES (?, ?, ?)",
                           (episode, step, pickle.dumps(step_data_blob)))
            self.collected_actions.append(action)
            self.collected_reward += reward

    def reset_episode_data(self):
        """Resets the collected data for a new episode."""
        self.collected_reward = 0
        self.collected_actions = []
        self.times = []

    def load_episode(self, episode):
        """Loads and returns the metadata for a specific episode."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT meta_data FROM episodes WHERE id = ?", (episode,))
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None

    def load_episode_step(self, episode, step):
        """Loads and returns the data for a specific step of an episode."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM steps WHERE episode_id = ? AND step = ?", (episode, step))
            row = cursor.fetchone()
            return pickle.loads(row[0]) if row else None

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

    def delete_test_folder(self):
        """Deletes the SQLite database file."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def store_own_episode(self, episode, meta_data, agent_selection):
        """
        Override this method to store the test result.
        """
        pass

    def store_own_episode_step(self, episode, step, state, action,  next_state, reward, done, truncated, info, step_data, agent_selection):
        """
        Override this method to store the test result.
        """
        pass