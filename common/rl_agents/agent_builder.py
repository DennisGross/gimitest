import os
from common.rl_agents.dqn_agent import *
'''
HOW TO ADD MORE AGENTS?
1) Create a new AGENTNAME.py with an AGENTNAME class
2) Inherit the agent-class
3) Override the methods
4) Import this py-script into this script
5) Add additional agent hyperparameters to the argparser
6) Add to build_agent the building procedure of your agent
'''
class AgentBuilder():

    @staticmethod
    def layers_neurons_to_number_of_neurons(layers, neurons):
        number_of_neurons = []
        for i in range(layers):
            number_of_neurons.append(neurons)
        return number_of_neurons

    @staticmethod
    def build_agent(command_line_arguments, gimi_env):
        #print('Build model with', model_root_folder_path, command_line_arguments)
        #print('Environment', observation_space.shape, action_space.n)
        model_root_folder_path = os.path.join("projects",command_line_arguments['project_name'], "model")
        # check if path exists
        if not os.path.exists(model_root_folder_path):
            os.mkdir(model_root_folder_path)
        try:
            state_dimension = gimi_env.observation_space.shape[0]
        except:
            state_dimension = 1
        agent = None
        if command_line_arguments['rl_algorithm'] == "dqn_agent":
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])
            agent = DQNAgent(state_dimension, number_of_neurons, gimi_env.action_space.n, epsilon=command_line_arguments['epsilon'], epsilon_dec=command_line_arguments['epsilon_dec'], epsilon_min=command_line_arguments['epsilon_min'], gamma=command_line_arguments['gamma'], learning_rate=command_line_arguments['lr'], replace=command_line_arguments['replace'], batch_size=command_line_arguments['batch_size'], replay_buffer_size=command_line_arguments['replay_buffer_size'])
            agent.load(model_root_folder_path)
        return agent