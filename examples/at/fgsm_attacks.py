import gymnasium as gym
import numpy as np
import torch
import tqdm


from gimitest.env_decorator import EnvDecorator


import sys
sys.path.append('../utils/')
from gtester import AdversarialTester
from common import load_dqn
from plotting import plot_fgsm_results


if __name__ == '__main__':
    torch.set_num_threads(1)

    epsilon = 0.0001
    env = gym.make('MountainCar-v0')
    model_path = 'models/dqn_mountain_car.zip'
    model = load_dqn(model_path)
    gtester = AdversarialTester(env, agents=model, epsilon=epsilon)
    EnvDecorator.decorate(env, gtester)

    # attacks the policy
    obs, _ = env.reset()
    budget = 10_000

    import tqdm
    for _ in tqdm.tqdm(range(budget)):
        action, _state = model.predict(obs, state=None, deterministic=True)
        obs, reward, terminated, truncated, _info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

    adversaries = np.array(gtester.adversaries)
    observations = np.array(gtester.observations)

    print('adversaries found:', len(adversaries))
    np.savetxt('adversaries.txt', adversaries, delimiter=',')
    np.savetxt('observations.txt', observations, delimiter=',')

    fig, ax = plot_fgsm_results(observations, adversaries)
    fig.savefig('test.png')