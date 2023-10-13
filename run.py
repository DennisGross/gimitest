from common.utilities.helper import *
from common.utilities.project import Project
import gym
from common.gimi.gym import GimiGym
from common.utilities.training import *
from common.configurators.configurator_builder import ConfiguratorBuilder
from common.test_cases.test_case_builder import TestCaseBuilder
# Ignore numpy warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



if __name__ == '__main__':
    print("=======================")
    args = get_arguments()
    set_random_seed(args['seed'])
    m_project = Project(args)
    m_project.load_saved_command_line_arguments()

    # Build environment
    env = gym.make(m_project.command_line_arguments['env'])
    gimi_env = GimiGym(env, m_project.command_line_arguments['state_attribute'], m_project.command_line_arguments['max_steps'])

    # Build Configurator
    configurator = ConfiguratorBuilder.build_configurator(m_project.command_line_arguments['configurator'], gimi_env)
    gimi_env.set_configurator(configurator)


    # Build agent
    m_project.create_agent(m_project.command_line_arguments, gimi_env)

    # Create test case
    m_test_case = TestCaseBuilder.build_test_case(m_project.command_line_arguments['test_case'], gimi_env, m_project.agent)

   


    if m_test_case is None:
        # Execute task
        execute(m_project, gimi_env, m_project.command_line_arguments['deploy'], m_project.command_line_arguments['num_episodes'])
    else:
        result = m_test_case.run(m_project, gimi_env)
        print(result)
