import random
from pettingzoo.classic import tictactoe_v3
import sys
sys.path.append('../gimitest')
from gym_decorator import GymDecorator
from test_cases.test_case import TestCase


class ReachabilityTestCase(TestCase):

    def __init__(self, parameters={}):
        super(ReachabilityTestCase, self).__init__(parameters)

    def step_execute(self, env, original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info, agent_selection):
        print(original_state, agent_selection, original_terminated or original_truncated)
        return original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info



env = tictactoe_v3.env(render_mode="human")

# Reachability test case
test_case = ReachabilityTestCase({"goal": 5})
test_cases = [test_case]
env = GymDecorator.decorate_gym(env, test_cases, None)

# Play 10 episodes of the game
for episode in range(3):
    print(f"Episode {episode + 1}")

    # Original environment setup for tictactoe_v3
    env.reset()

    for agent in env.agent_iter():
        observation, _, done, truncated, _ = env.last()
        if done or truncated:
            action = None
            break
        else:
            action = env.action_space(agent).sample()
        print(env.infos)
        env.step(action)
        env.render()

        
    print("End of Episode\n\n")
env.close()
