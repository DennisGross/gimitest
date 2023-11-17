import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random

class TestAnalyse:

    def __init__(self, test_logger, parameters = {}):
        self.test_logger = test_logger
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

    

    def plot_key_value_over_episodes(self, key, filepath=None, xlabel="episode"):
        number_of_episodes = self.test_logger.count_episodes()
        values = []
        for episode in range(1, number_of_episodes):
            try:
                episode_dict = self.test_logger.load_episode(episode)
                value = self.__find_value_of_key_in_dictionary(episode_dict, key)
                values.append(value)
            except Exception as e:
                print(e)
                continue

        plt.boxplot(values)  # Changed from plt.plot to plt.boxplot
        plt.ylabel(key)
        plt.xlabel(xlabel)
        plt.grid(True)  # Modified for better visibility of grid
        if filepath is None:
            plt.savefig(key + '.png')
        else:
            plt.savefig(filepath)
        plt.clf()

    def plot_keys_over_episodes(self, keys, filepath=None, xlabel="episode", ylbale="value"):
        number_of_episodes = self.test_logger.count_episodes()
        values = {}
        for episode in range(1, number_of_episodes):
            try:
                episode_dict = self.test_logger.load_episode(episode)
                for key in keys:
                    value = self.__find_value_of_key_in_dictionary(episode_dict, key)
                    if key in values:
                        values[key].append(value)
                    else:
                        values[key] = [value]
                
            except Exception as e:
                print(e)
                continue

        # Line plot
        line_styles = ['-', '--', '-.', ':']  # Define different line styles
        markers = ['o', '^', 's', 'D', '*', 'x', '+', 'p']

        style_index = 0  # To track the current line style
        marker_index = 0

        for key in keys:
            plt.plot(values[key], label=key, linestyle=line_styles[style_index], marker=markers[marker_index])
            style_index = (style_index + 1) % len(line_styles)  # Cycle through styles
            marker_index = (marker_index + 1) % len(markers)  # Cycle through styles


        plt.ylabel(ylbale)
        plt.xlabel(xlabel)
        plt.grid(True)  # Modified for better visibility of grid
        plt.legend()
        if filepath is None:
            plt.savefig(key + '.png')
        else:
            plt.savefig(filepath)
        plt.clf()




    def plot_key1_key2_and_value(self, key1, key2, value_key, xlabel=None, ylabel=None, filepath=None):
        number_of_episodes = self.test_logger.count_episodes()
        para1 = []
        para2 = []
        values = []
        for episode in range(1, number_of_episodes):
            try:
                episode_dict = self.test_logger.load_episode(episode)
                print(episode_dict)
                value = self.__find_value_of_key_in_dictionary(episode_dict, value_key)
                #print(value)
                values.append(value)
                para1.append(self.__find_value_of_key_in_dictionary(episode_dict, key1))
                para2.append(self.__find_value_of_key_in_dictionary(episode_dict, key2))
            except Exception as e:
                print(e)
                continue
        # plot scatter plot with x = para1, y = para2, color = heat values, and alpha=0.5
        plt.scatter(para1, para2, c=values, cmap='viridis', alpha=0.5)
        
        plt.colorbar()
        if xlabel is None:
            plt.xlabel(key1)
        else:
            plt.xlabel(xlabel)
        if ylabel is None:
            plt.ylabel(key2)
        else:
            plt.ylabel(ylabel)
        plt.grid(0.2)
        if filepath == None:
            plt.savefig(key1 + '_' + key2 + '_' + value_key + '.png')
        else:
            plt.savefig(filepath)
        
        plt.clf()

    def plot_action_distribution(self, filepath=None):
        number_of_episodes = self.test_logger.count_episodes()
        actions = []
        for episode in range(1, number_of_episodes):
            try:
                number_of_steps = self.test_logger.count_episode_steps(episode)
                for step in range(0, number_of_steps):
                    action = str(self.test_logger.load_episode_step(episode, step)["action"])
                    actions.append(action)
                
            except Exception as e:
                print(e)
                continue
        # Group strings in list and count their occurences
        actions = [[x,actions.count(x)] for x in set(actions)]
        
        # Plot distribution in histogram
        plt.bar([x[0] for x in actions], [x[1] for x in actions])
        plt.ylabel('number of actions')
        plt.xlabel('action')
        #plt.grid(0.2)
        # only horizontal grid
        plt.gca().yaxis.grid(True)
        if filepath == None:
            plt.savefig('action_distribution.png')
        else:
            plt.savefig(filepath)
        plt.clf()

    
    def plot_state_action_behaviour(self, filepath=None):
        states = []
        actions = []
        number_of_episodes = self.test_logger.count_episodes()
        for episode in range(1, number_of_episodes):
            try:
                number_of_steps = self.test_logger.count_episode_steps(episode)
                for step in range(0, number_of_steps):
                    state = self.test_logger.load_episode_step(episode, step)["state"]
                    action = int(str(self.test_logger.load_episode_step(episode, step)["action"]).replace(",", "").replace("(", "").replace(")", ""))
                    states.append(state)
                    actions.append(action)
            except Exception as e:
                print(e)
                continue
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.9, random_state=42)
        
        # Train a decision tree classifier
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model Accuracy: {accuracy * 100:.2f}%')
        
        # Plot the tree with better resolution and annotations
        fig, ax = plt.subplots(figsize=(40, 20))  # Increase figure size for better resolution
        tree.plot_tree(clf, ax=ax, filled=True, fontsize=10)
        plt.title('State-Action Behavior Tree')
        if filepath == None:
            plt.savefig('state_action_behaviour_high_res.png', dpi=300)  # Save with higher DPI for better resolution
        else:
            plt.savefig(filepath, dpi=300)
        plt.clf()

    def plot_state_reward_map(self, filepath=None):
        states = []
        rewards = []
        number_of_episodes = self.test_logger.count_episodes()
        for episode in range(1, number_of_episodes):
            try:
                number_of_steps = self.test_logger.count_episode_steps(episode)
                for step in range(0, number_of_steps):
                    state = self.test_logger.load_episode_step(episode, step)["state"]
                    reward = self.test_logger.load_episode_step(episode, step)["reward"]
                    states.append(state)
                    rewards.append(reward)
            except Exception as e:
                print(e)
                continue

        # Sample data: states as a numpy array of shape (n_samples, n_features)
        # rewards as a numpy array of shape (n_samples, )
        states = np.array(states)
        rewards = np.array(rewards)

        
        # Load your own 'states' and 'rewards' data here

        # Perform PCA
        pca = PCA(n_components=2)
        states_pca = pca.fit_transform(states)

        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        states_tsne = tsne.fit_transform(states)

        # Create plots
        fig, axs = plt.subplots(1, 1)

        # PCA Plot
        sc = axs.scatter(states_pca[:, 0], states_pca[:, 1], c=rewards, cmap='viridis')
        axs.set_title("PCA")
        axs.set_xlabel("First Principal Component")
        axs.set_ylabel("Second Principal Component")
        plt.colorbar(sc, ax=axs)

        if filepath == None:
            plt.savefig('pca_tsne.png')
        else:
            plt.savefig(filepath)

        plt.clf()

