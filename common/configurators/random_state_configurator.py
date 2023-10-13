from common.configurators.configurator import Configurator
import random
import math

class RandomInitStateConfigurator(Configurator):

    def __init__(self, env, parameter_str, state_str):
        super().__init__(env, parameter_str, state_str)


    def generate_state(self, state):
        #print(len(self.env_state_dict.keys()))
        #print(self.env_state_dict)
        # Generate random state
        for idx_key in self.env_state_dict.keys():
            # if inf or -inf, continue
            if self.THRESHOLD * -1 > self.env_state_dict[idx_key]['low'] or self.THRESHOLD < self.env_state_dict[idx_key]['high']:
                # If inf or -inf, continue
                continue
            if self.env_state_dict[idx_key]['type'] == 'int':
                state[idx_key] = random.randint(self.env_state_dict[idx_key]['low'], self.env_state_dict[idx_key]['high'])
            elif self.env_state_dict[idx_key]['type'] == 'float':
                state[idx_key] = random.uniform(self.env_state_dict[idx_key]['low'], self.env_state_dict[idx_key]['high'])
            elif self.env_state_dict[idx_key]['type'] == 'bool':
                state[idx_key] = random.choice([True, False])
            else:
                raise NotImplementedError("Type not implemented")
        return state