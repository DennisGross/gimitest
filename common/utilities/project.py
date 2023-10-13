import os
from common.rl_agents.agent_builder import AgentBuilder
import time
class Project():

    def __init__(self, command_line_arguments):
        self.command_line_arguments = command_line_arguments
        # Check if projects folder exist
        if not os.path.exists("projects"):
            os.mkdir("projects")
        self.project_folder_path = os.path.join("projects",self.command_line_arguments['project_name'])
        
    

    def create_agent(self, command_line_arguments, gimi_gym):
        # Build agent with the model and the hyperparameters
        self.agent = AgentBuilder.build_agent(command_line_arguments, gimi_gym)
        return self.agent

    def save(self, episode,best_reward_of_sliding_window):
        self.agent.save(os.path.join("projects",self.command_line_arguments['project_name'], "model"))
        # Save command line dictionary to json
        self.__save_dictionary_to_json(self.command_line_arguments, os.path.join("projects",self.command_line_arguments['project_name'], "command_line_arguments.json"))
        # Add best_reward_of_sliding_window to txt file best_reward_of_sliding_window.txt
        with open(os.path.join("projects",self.command_line_arguments['project_name'], "best_reward_of_sliding_window.txt"), 'a') as fp:
            fp.write(str(time.time()) + "," + str(episode) + "," +str(best_reward_of_sliding_window) + "\n")


    def __load_dictionary_from_json(self, json_file_path):
        import json
        with open(json_file_path, 'r') as fp:
            return json.load(fp)

    def __save_dictionary_to_json(self, dictionary, json_file_path):
        import json
        with open(json_file_path, 'w') as fp:
            json.dump(dictionary, fp)
        

    def load_saved_command_line_arguments(self):
        # Check if project_name exists in projects
        
        if not os.path.exists(self.project_folder_path):
            os.mkdir(self.project_folder_path)

        if not os.path.exists(os.path.join(self.project_folder_path, "command_line_arguments.json")):
            return self.command_line_arguments
        
        saved_command_line_arguments = self.__load_dictionary_from_json(os.path.join(self.project_folder_path, "command_line_arguments.json"))
        if saved_command_line_arguments != None:
            old_task = saved_command_line_arguments['task']
            try:
                del saved_command_line_arguments['task']
            except:
                pass
            try:
                del saved_command_line_arguments['project_name']
            except:
                pass
            if self.command_line_arguments['preprocessor'] != '':
                # Only delete it if it is not set by the command line (in this case take new one)
                # If "None", later during building none will be created
                try:
                    del saved_command_line_arguments['preprocessor']
                except:
                    pass
            try:
                del saved_command_line_arguments['epsilon']
            except:
                pass
            try:
                del saved_command_line_arguments['epsilon_dec']
            except:
                pass
            try:
                del saved_command_line_arguments['epsilon_min']
            except:
                pass
            try:
                del saved_command_line_arguments['seed']
            except:
                pass
            try:
                del saved_command_line_arguments['deploy']
            except:
                pass
            try:
                del saved_command_line_arguments['training_threshold']
            except:
                pass
            try:
                del saved_command_line_arguments['num_episodes']
            except:
                pass
            try:
                del saved_command_line_arguments['eval_interval']
            except:
                pass
            for key in saved_command_line_arguments.keys():
                self.command_line_arguments[key] = saved_command_line_arguments[key]