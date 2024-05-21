import gymnasium as gym
import numpy as np
import torch
import tqdm

from stable_baselines3 import PPO
from gimitest.env_decorator import EnvDecorator


import sys
sys.path.append('../utils/')
from logger import Logger
from gtester import InitialForceTester, LL_MIN, LL_MAX
from common import load_ppo


def random_testing(model: PPO, env_seed: int = 0, n: int = 1000, logpath: str = 'random_testing.txt'):
    env: gym.Env = gym.make('LunarLander-v2')
    gtester = InitialForceTester(env, env_seed)
    EnvDecorator.decorate(env, gtester)

    rng = np.random.default_rng(0)
    inputs = rng.uniform(low=LL_MIN, high=LL_MAX, size=(n, 2))

    logger = Logger(logpath, columns=['input', 'reward', 'oracle', 'episode_length'])
    logger.write_columns()

    for i in tqdm.tqdm(range(len(inputs))):
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

        logger.log(
            input=input,
            reward=acc_reward,
            oracle=(reward==-100),
            episode_length=t
        )

    env.close()
    return gtester.action_distribution


if __name__ == '__main__':
    torch.set_num_threads(1)

    model_path = '../models/ppo_lunar_lander.zip'
    model = load_ppo(model_path)

    action_distribution = random_testing(model, 0, 10_000)
    np.savetxt('random_testing_action_distribution.txt', action_distribution)
