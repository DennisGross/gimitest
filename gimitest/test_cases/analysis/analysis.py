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