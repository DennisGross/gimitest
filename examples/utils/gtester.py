import torch
import numpy as np

from typing import List, Union
from gimitest.gtest import GTest
from stable_baselines3 import PPO, DQN


LL_MIN = -1000
LL_MAX = 1000
CP_MIN = -0.05
CP_MAX = 0.05


class InitialForceTester(GTest):

    def __init__(self, env, env_seed: int = 0, agents=None):
        super().__init__(env, agents)
        assert env_seed >= 0
        self.env_seed = env_seed
        self.initial_force: np.ndarray = np.array([0.0, 0.0])
        self.action_distribution = np.zeros(env.action_space.n)


    def pre_reset_configuration(self):
        return {'seed': self.env_seed}


    def post_reset_configuration(self, next_state):
        env = self.env.unwrapped
        initial_force = tuple(self.initial_force)

        env = self.env.unwrapped
        env.lander.ApplyForceToCenter(
            initial_force,
            True,
        )

        return env.step(0)[0]

    def post_step_test(self, state, action, next_state, reward, terminated, truncated, info, agent_selection):
        self.action_distribution[action] += 1
        return state, action, next_state, reward, terminated, truncated, info


    def set_initial_force(self, force: np.ndarray):
        assert len(force) == 2
        self.initial_force = force.astype(float).copy()


class AdversarialTester(GTest):

    def __init__(self, env, agents: DQN = None, epsilon: float = 0.0001):
        super().__init__(env, agents)
        self.observations = []
        self.adversaries = []
        self.epsilon = epsilon


    def post_step_test(self, state, action, next_state, reward, terminated, truncated, info, agent_selection):
        self.observations.append(state.copy())
        adversary = self._fsgm(self.agents, state, action[0], self.epsilon)

        new_action, _state = self.agents.predict(adversary, state=None, deterministic=True)

        if not np.array_equal(action[0], new_action):
            self.adversaries.append(adversary.copy())

        return state, action, next_state, reward, terminated, truncated, info


    def _fsgm(self, model: Union[PPO, DQN], obs: np.ndarray, action: np.ndarray, epsilon: float) -> np.ndarray:
        '''Returns the adversary of given state @obs and @epsilon.'''
        tobs = torch.as_tensor(obs[None])
        tobs.requires_grad = True

        if isinstance(model, PPO):
            action_distribution = model.policy.get_distribution(tobs).distribution.probs
        else:
            q_values = model.q_net.forward(tobs)
            action_distribution = torch.nn.functional.softmax(q_values, dim=1)

        loss = torch.nn.functional.cross_entropy(
            action_distribution,
            torch.as_tensor(action[None])
        )

        model.policy.zero_grad()
        loss.backward()

        gradient = tobs.grad.data
        adversary = tobs + epsilon * gradient.sign()
        return adversary.detach()[0].cpu().numpy()


class InitialStateTester(GTest):

    def __init__(self, env, agents=None, parameters=...):
        super().__init__(env, agents, parameters)
        self.initial_state: np.ndarray = np.zeros(env.observation_space.shape[0], dtype=np.float32)


    def post_reset_configuration(self, next_state):
        # deterministic executions by setting the unwrapped environment
        self.set_attribute(self.env.unwrapped, 'state', np.array(self.initial_state, dtype=np.float32))
        return np.array(self.initial_state, dtype=np.float32)


    def set_initial_state(self, state: np.ndarray):
        assert len(state) == len(self.initial_state)
        assert np.all(state >= CP_MIN), state
        assert np.all(state <= CP_MAX), state
        self.initial_state = np.array(state, dtype=np.float32)