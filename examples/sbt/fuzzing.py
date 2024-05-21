import gymnasium as gym
import numpy as np
import torch
import tqdm
import time

from gimitest.env_decorator import EnvDecorator
from typing import List, Tuple, Any

import sys
sys.path.append('../utils/')
from gtester import InitialForceTester, LL_MIN, LL_MAX
from common import load_ppo
from mdpfuzz import Fuzzer, Executor


MUTATION_INTENSITY = 5.0


class LunarLanderExecutor(Executor):

    def __init__(self, sim_steps, env_seed) -> None:
        super().__init__(sim_steps, env_seed)
        self.env: gym.Env = gym.make('LunarLander-v2')
        self.gtester = InitialForceTester(self.env, self.env_seed)
        EnvDecorator.decorate(self.env, self.gtester)

    def generate_input(self, rng: np.random.Generator):
        '''Generates a single input between the given bounds (parameters).'''
        return rng.uniform(low=LL_MIN, high=LL_MAX, size=2)


    def generate_inputs(self, rng: np.random.Generator, n: int = 1):
        '''Generates @n inputs with the lower and upper bounds parameters.'''
        if n == 1:
            return self.generate_input(rng)
        else:
            return rng.uniform(low=LL_MIN, high=LL_MAX, size=(n, 2))


    def mutate(self, input: List[np.ndarray], rng: np.random.Generator, **kwargs):
        return np.clip(
            rng.normal(input, MUTATION_INTENSITY),
            LL_MIN,
            LL_MAX
            )


    def load_policy(self):
        return load_ppo('../models/ppo_lunar_lander.zip')


    def execute_policy(self, input: np.ndarray, policy: Any) -> Tuple[float, bool, np.ndarray, float]:
        '''Executes the model and returns the trajectory data. Useful for MDPFuzz.'''
        t0 = time.time()
        self.gtester.set_initial_force(input)
        obs, _info = self.env.reset()
        acc_reward = 0
        state = None
        t = 0
        obs_seq = []
        while True:
            t += 1
            obs_seq.append(obs)
            action, state = policy.predict(obs, state=state, deterministic=True)
            obs, reward, terminated, truncated, _info = self.env.step(action)
            acc_reward += reward
            if terminated or truncated:
                break

        return acc_reward, (reward == -100), np.array(obs_seq), time.time() - t0


if __name__ == '__main__':
    torch.set_num_threads(1)

    # tests executor class
    # rng = np.random.default_rng(0)
    # executor: Executor = LunarLanderExecutor(1000, 0)
    # input = executor.generate_input(rng)
    # policy = executor.load_policy()
    # reward, oracle, sequence, exec_time = executor.execute_policy(input, policy)
    # print(input, reward, oracle, exec_time)
    # print(sequence.shape)

    executor = LunarLanderExecutor(sim_steps=1000, env_seed=0)
    model = executor.load_policy()
    fuzzer = Fuzzer(random_seed=0, executor=executor)

    fuzzer.fuzzing(
        n=1_000,
        policy=model,
        test_budget=10_000,
        saving_path='fuzzing',
        local_sensitivity=True,
        exp_name='Lunar Lander'
        )

    np.savetxt('fuzzing_action_distribution.txt', executor.gtester.action_distribution)