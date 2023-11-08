from pettingzoo.butterfly import pistonball_v6
import random
from pettingzoo.classic import tictactoe_v3
import sys
sys.path.append('../gimitest')
from gym_decorator import GymDecorator
from test_cases.test_case import TestCase


class ReachabilityTestCase(TestCase):

    def __init__(self, parameters={}):
        super(ReachabilityTestCase, self).__init__(parameters)
        self.counter = 0

    def step_execute(self, env, original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info, agent_selection):
        print(self.counter)
        self.counter += 1
        return original_state, action_args, original_next_state, original_reward, original_terminated, original_truncated, original_info

env = pistonball_v6.parallel_env(render_mode="human")
# Reachability test case
test_case = ReachabilityTestCase({"goal": 5})
test_cases = [test_case]
env = GymDecorator.decorate_gym(env, test_cases, None)

observations, infos = env.reset(seed=42)


steps = 0

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    steps+=1
    if steps > 25:
        break
env.close()