import random
from pettingzoo.classic import tictactoe_v3
import sys
sys.path.append('../gimitest')
from gym_decorator import GymDecorator
from test_cases.test_case import TestCase
from test_cases.loggers.test_logger import TestLogger
from test_cases.test_case_decorator import TestCaseDecorator
from test_cases.analysis.analysis import TestAnalyse
import numpy as np

class SarsaMaxAgent():

    def __init__(self, number_of_actions, epsilon=0.5, epsilon_dec=0.9999, epsilon_min=0.01, alpha=0.6, gamma = 0.9):
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.Q = dict()



    def select_action(self, state, deploy=False):
        state, next_state =  self.__make_sure_that_states_in_dictionary(state)
        action_index = None
        if deploy == True:
            action_index = np.argmax(self.Q[str(state)])
            return int(action_index)
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_dec
        if np.random.rand() < self.epsilon:
            # Random Choice
            action_index = np.random.choice(self.number_of_actions)
        else:
            # Greedy Choice
            action_index = np.argmax(self.Q[str(state)])
        return int(action_index)


    def __make_sure_that_states_in_dictionary(self, state, next_state=None):
        state = str(state)
        if state not in self.Q.keys():
            self.Q[state] = [0]*self.number_of_actions
        if next_state is not None:
            next_state = str(next_state)
            if next_state not in self.Q.keys():
                self.Q[next_state] = [0]*self.number_of_actions
        return state, next_state

    def store_experience(self, state, action, reward, next_state, terminal):
        '''
        Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - terminal: whether the episode is complete (True or False)
        '''
        state, next_state = self.__make_sure_that_states_in_dictionary(state, next_state)
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * max(self.Q[next_state]) - self.Q[state][action])




class ReachabilityTestCase(TestCase):

    def __init__(self, parameters={}):
        super(ReachabilityTestCase, self).__init__(parameters)

    def step_execute(self, env, original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info, agent_selection):
        # Check if an agent won the game
        success = None
        if original_terminated or original_truncated:
            if original_reward == 1:
                success = agent_selection
            else:
                success = 0

        self.meta_data["success"] = reward
        
        return original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info





env = tictactoe_v3.env()

agent1 = SarsaMaxAgent(env.action_spaces["player_1"].n)
agent2 = SarsaMaxAgent(env.action_spaces["player_2"].n)


m_logger = TestLogger("tic_tac_toe")
m_test_case = ReachabilityTestCase()
m_test_case = TestCaseDecorator.decorate_test_case_with_test_logger(m_test_case, m_logger)
test_cases = [m_test_case]

env = GymDecorator.decorate_gym(env, test_cases, None)


# Play 10 episodes of the game
for episode in range(10):
    # Original environment setup for tictactoe_v3
    env.reset()
    state, reward, done, truncated, _ = env.last()
    for agent in env.agent_iter():
        if done or truncated:
            action = None
            break
        else:
            if agent == "player_1":
                action = agent1.select_action(state)
            else:
                action = agent2.select_action(state)

        env.step(action)
        next_state, reward, done, truncated, _ = env.last()
        if agent == "player_1":
            agent1.store_experience(state, action, reward, next_state, done)
        else:
            agent2.store_experience(state, action, reward, next_state, done)
        state = next_state
        env.render()

        
    print("End of Episode\n\n")
env.close()


m_test_analyse = TestAnalyse(m_logger)
m_test_analyse.plot_key_value_over_episodes("success", filepath="tic_tac_toe_success.png", xlabel="")
m_test_analyse.plot_key_value_over_episodes("entropy_of_actions", filepath="tic_tac_toe_entropy.png", xlabel="")

#m_logger.delete_test_folder()