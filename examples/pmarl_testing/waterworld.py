"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from pettingzoo.sisl import waterworld_v4

import sys
sys.path.append('../../gimitest')
from env_decorator import EnvDecorator
from gtest import GTest
from glogger import GLogger
from gtest_decorator import GTestDecorator
import random

class WaterTester(GTest):

    def pre_reset_configuration(self):
        env = self.env.unwrapped
        rnd_n_evaders = random.randint(1, 10)
        rnd_n_n_poisons = random.randint(1, 20)
        env.n_evaders = rnd_n_evaders
        env.n_poisons = rnd_n_n_poisons
        self.episode_data["n_evaders"] = rnd_n_evaders
        self.episode_data["n_poisons"] = rnd_n_n_poisons

    def post_step_test(self, state, action, next_state, reward, terminated, truncated, info, agent_selection):
        return state, action, next_state, reward, terminated, truncated, info


def train_butterfly_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn.parallel_env(**env_kwargs)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        learning_rate=1e-3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    model.save(f"../models/waterworld.zip")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, gtesting=False, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.parallel_env(**env_kwargs)
    m_gtest = None
    if gtesting:
        m_gtest = WaterTester(env)
        m_glogger = GLogger("waterworld")
        m_gtest = GTestDecorator.decorate_with_logger(m_gtest, m_glogger)
        env = EnvDecorator.decorate(env, m_gtest)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"../models/waterworld.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    # Evaluate the trained model
    rewards = []
    for i in range(num_games):
        print(f"Game {i+1}/{num_games}")
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            actions = {}
            #print(common_obs)
            for agent in env.agents:
                actions[agent] = model.predict(obs[agent])[0]
            obs, reward, termination, truncation, info = env.step(actions)
            # Sum the rewards that each agent received
            episode_reward += sum(reward.values())
            done = all(termination.values()) and all(truncation.values())
        rewards.append(episode_reward)

    

    avg_reward = sum(rewards) / len(rewards)
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    if gtesting:
        df = m_glogger.create_episode_dataset(["n_evaders", "n_poisons", "collected_reward"])
        df.to_csv("waterworld_evaders_reward.csv")
        m_glogger.delete_database()
        print(df.head(n=100))
        # Scatter plot
        import matplotlib.pyplot as plt
        plt.scatter(df["n_evaders"], df["n_poisons"], c=df["collected_reward"])
        plt.xlabel("Number of evaders")
        plt.ylabel("Number of poisons")
        # Colorbar
        cbar = plt.colorbar()
        cbar.set_label("Collected reward")
        #plt.title("Waterworld evaders reward")
        plt.tight_layout()
        plt.grid(alpha=0.2)
        plt.savefig("waterworld_evaders_reward.png")
        plt.savefig("waterworld_evaders_reward.eps")
    
    return avg_reward


if __name__ == "__main__":
    testing_flag = False  # Set this flag to control training/testing
    NUMBER_OF_TESTS = 100
    env_fn = waterworld_v4
    
    env_kwargs = {}

    

    if testing_flag:
        # Only evaluate
        eval(env_fn, num_games=NUMBER_OF_TESTS, render_mode=None, gtesting=True, **env_kwargs)
    else:
        # Train and then evaluate
        train_butterfly_supersuit(env_fn, steps=196_608, seed=0, **env_kwargs)
        eval(env_fn, num_games=NUMBER_OF_TESTS, render_mode=None, gtesting=True, **env_kwargs)
