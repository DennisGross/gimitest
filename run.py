from common.utilities.helper import *
from common.utilities.project import Project
import gym
from common.gimi.gym import GimiGym
from common.utilities.training import *


if __name__ == '__main__':
    print("=======================")
    args = get_arguments()
    set_random_seed(args['seed'])
    m_project = Project(args)
    m_project.load_saved_command_line_arguments()

    env = gym.make(m_project.command_line_arguments['env'])
    gimi_env = GimiGym(env, m_project.command_line_arguments['state_attribute'], m_project.command_line_arguments['max_steps'])

    m_project.create_agent(m_project.command_line_arguments, gimi_env)
    execute(m_project, gimi_env, m_project.command_line_arguments['deploy'])