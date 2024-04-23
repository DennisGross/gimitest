from gimitest.gtest import GTest
import random

class RandomSearchBasedStateIndependentTesting(GTest):

    def __init__(self, env, agents=None, parameters={}):
        # keys: environment parameters needs to be accessable via env.unwrapped
        # parameter : {lower_bound: value, upper_bound: value, precision(optional): value, type: integer, float, bool}
        super().__init__(env, agents, parameters)

    def pre_reset_configuration(self):
        
        for parameter, value in self.parameters.items():
            if value["type"] == "int":
                # Get random value between lower and upper bound
                random_value = random.randint(value["lower_bound"], value["upper_bound"])
            elif value["type"] == "float":
                # Get random value between lower and upper bound and round to closes step
                random_value = random.uniform(value["lower_bound"], value["upper_bound"])
                # Check if step is defined
                if "precision" in value:
                    random_value = round(random_value / value["precision"]) * value["precision"]
            elif value["type"] == "bool":
                # Get random boolean value
                random_value = random.choice([True, False])
            # Set the value of the parameter in the environment
            self.env.unwrapped.__setattr__(parameter, random_value)
            self.episode_data[parameter] = random_value
            
    
