import gymnasium as gym
import numpy as np
import torch
import tqdm


from stable_baselines3 import PPO
from gimitest.env_decorator import EnvDecorator
from typing import List


import sys
sys.path.append('../utils/')
from logger import Logger
from gtester import InitialForceTester, LL_MIN, LL_MAX
from common import load_ppo


MUTATION_INTENSITY = 5.0


def evolutionary_search(
    model: PPO,
    env_seed: int,
    num_iter: int,
    pop_size: int,
    rng: np.random.Generator,
    logpath: str
    ):

    logger = Logger(logpath, columns=['input', 'reward', 'oracle', 'episode_length'])
    logger.write_columns()

    env: gym.Env = gym.make('LunarLander-v2')
    gtester = InitialForceTester(env, env_seed)
    EnvDecorator.decorate(env, gtester)


    def _mutate(inputs: List[np.ndarray], rng: np.random.Generator):
        return [
            np.clip(
                rng.normal(x, MUTATION_INTENSITY),
                LL_MIN,
                LL_MAX
                )
            for x in inputs
            ]


    def _evaluate(inputs: np.ndarray, model: PPO, logger: Logger = None):
        acc_rewards = []

        for i in range(len(inputs)):
            input = inputs[i]
            gtester.set_initial_force(input)
            obs, _info = env.reset()
            acc_reward = 0
            state = None
            t = 0
            while True:
                t += 1
                action, state = model.predict(obs, state=state, deterministic=True)
                obs, reward, terminated, truncated, _info = env.step(action)
                acc_reward += reward
                if terminated or truncated:
                    break

            acc_rewards.append(acc_reward)

            if logger is not None:
                logger.log(
                    input=input,
                    reward=acc_reward,
                    oracle=(reward==-100),
                    episode_length=t
                )
        return acc_rewards

    # initial random population
    pop = rng.uniform(low=LL_MIN, high=LL_MAX, size=(pop_size, 2))
    pop_scores = _evaluate(pop, model, logger)
    for i in tqdm.tqdm(range(num_iter)):
        # generates offspring
        offspring = _mutate(pop, rng)
        # evaluates the offspring
        offspring_scores = _evaluate(offspring, model, logger)

        # selects the most performing individuals to form the new population
        joined_pop = np.vstack([pop, offspring])
        joined_scores = np.hstack([pop_scores, offspring_scores])
        median_score = np.median(joined_scores)

        # print(i, median_score)

        mask = (joined_scores <= median_score)
        pop = joined_pop[mask].copy()
        pop_scores = joined_scores[mask]

        if len(pop) > pop_size:
            pop = pop[:pop_size]
            pop_scores = pop_scores[:pop_size]

    env.close()
    return gtester.action_distribution

if __name__ == '__main__':
    torch.set_num_threads(1)

    model_path = '../models/ppo_lunar_lander.zip'
    model = load_ppo(model_path)

    num_iter = 100
    pop_size = 100
    rng = np.random.default_rng(42)
    action_distribution = evolutionary_search(model, 0, num_iter, pop_size, rng, 'es.txt')
    np.savetxt('es_action_distribution.txt', action_distribution)