import gymnasium as gym
import numpy as np
import torch
import tqdm

from stable_baselines3 import DQN
from gimitest.env_decorator import EnvDecorator


import sys
sys.path.append('../utils/')
from gtester import InitialStateTester, CP_MAX, CP_MIN
from common import load_dqn


def metamorphic_testing(model: DQN, n: int = 1000):
    env: gym.Env = gym.make('CartPole-v1')
    gtester = InitialStateTester(env)
    EnvDecorator.decorate(env, gtester)

    rng = np.random.default_rng(0)
    inputs = rng.uniform(low=CP_MIN, high=CP_MAX, size=(n, 4))

    bugs = []

    def _execute(env: gym.Env, model: DQN, timesteps: int = 400):
        obs, _info = env.reset()
        acc_reward = 0
        state = None
        for _ in range(timesteps):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            acc_reward += reward
            if terminated or truncated:
                break
        return acc_reward, (acc_reward != timesteps)

    def _unrelaxed(state: np.ndarray, intensity: float = 0.01):
        angle = state[2]

        if angle > 0:
            new_angle = np.clip(
                angle + rng.uniform(low=0.0, high=intensity),
                0.0,
                CP_MAX
            )
        else:
            new_angle = np.clip(
                angle - rng.uniform(low=0.0, high=intensity),
                CP_MIN,
                0.0
            )

        assert np.abs(new_angle) > np.abs(angle), f'{angle} -> {new_angle}'
        harder_state = state.copy()
        harder_state[2] = new_angle
        return harder_state


    for i in tqdm.tqdm(range(len(inputs))):
        input = inputs[i]
        gtester.set_initial_state(input)

        _acc_reward, relaxed_failure = _execute(env, model)

        # if failure, check that it solves an easier test case
        if relaxed_failure:
            harder_input = _unrelaxed(input)
            gtester.set_initial_state(harder_input)
            _, failure = _execute(env, model)
            if failure == False:
                bugs.append(input.copy())

    env.close()
    return inputs, np.array(bugs)


if __name__ == '__main__':
    torch.set_num_threads(1)

    model_path = '../models/dqn_cart_pole.zip'
    model = load_dqn(model_path)

    inputs, bugs = metamorphic_testing(model, n=10_000)
    print(len(bugs), 'found.')
    np.savetxt('cp_bugs.txt', bugs, delimiter=',')
    np.savetxt('cp_inputs.txt', inputs, delimiter=',')