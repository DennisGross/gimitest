from common.test_cases.test_case import TestCase
from common.test_cases.simple_state_reacher import SimpleStateReacher


class TestCaseBuilder():

    @staticmethod
    def build_test_case(configurator_str, gimi_env, rl_agent):
        config_name = configurator_str.split(";")[0]
        test_case = None
        if config_name == "simple_state_reacher":
            test_case = SimpleStateReacher(configurator_str)
        elif config_name == "":
            pass
        else:
            raise NotImplementedError("Configurator not implemented")
        
        return test_case