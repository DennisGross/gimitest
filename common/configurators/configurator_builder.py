from common.configurators.configurator import Configurator
from common.configurators.random_state_configurator import RandomInitStateConfigurator


class ConfiguratorBuilder():

    @staticmethod
    def build_configurator(configurator_str, gimi_env):
        config_name = configurator_str.split(";")[0]
        configurator = None
        if config_name == "random_state_configurator":
            configurator = RandomInitStateConfigurator(gimi_env, "", "")
        elif config_name == "":
            pass
        else:
            raise NotImplementedError("Configurator not implemented")
        
        return configurator