import matplotlib.pyplot as plt
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

    

    def plot_key_value_over_episodes(self, key):
        number_of_episodes = self.test_logger.count_episodes()
        values = []
        for episode in range(1, number_of_episodes):
            try:
                episode_dict = self.test_logger.load_episode(episode)
                print(episode_dict)
                value = self.__find_value_of_key_in_dictionary(episode_dict, key)
                print(value)
                values.append(value)
            except Exception as e:
                print(e)
                continue
        
        plt.plot(values)
        plt.ylabel(key)
        plt.xlabel('episode')
        plt.grid(0.2)
        plt.savefig(key + '.png')
        plt.clf()


    def plot_key1_key2_and_value(self, key1, key2, value_key, xlabel=None, ylabel=None):
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
        plt.savefig(key1 + '_' + key2 + '_' + value_key + '.png')
        plt.clf()


    