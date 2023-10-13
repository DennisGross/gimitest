from common.utilities.project import Project
import sys
from common.utilities.helper import *
import random
import math
import numpy as np
import torch
from collections import deque
import gc



def execute(project, env, deploy):
    all_episode_rewards = deque(maxlen=project.command_line_arguments['sliding_window_size'])
    all_property_results = deque(maxlen=project.command_line_arguments['sliding_window_size'])
    best_reward_of_sliding_window = math.inf * -1
    satisfied = False

    project.agent.load_env(env)
    try:
        for episode in range(project.command_line_arguments['num_episodes']):
            state = env.reset()
            done = False
            episode_reward = 0
            while done == False:
                action = project.agent.select_action(state, deploy)
                next_state, reward, done, info = env.step(action)
                if deploy==False:
                    project.agent.store_experience(state, action, reward, next_state, done)
                    project.agent.step_learn()
                state = next_state
                episode_reward+=reward
            # Log rewards
            all_episode_rewards.append(episode_reward)
            reward_of_sliding_window = np.mean(list(all_episode_rewards))

            if deploy==False:
                project.agent.episodic_learn()


            # Update best sliding window value
            if reward_of_sliding_window  > best_reward_of_sliding_window and len(all_episode_rewards)>=project.command_line_arguments['sliding_window_size']:
                best_reward_of_sliding_window = reward_of_sliding_window
                if deploy==False:
                    project.save(episode, best_reward_of_sliding_window)

                if (best_reward_of_sliding_window >= project.command_line_arguments['training_threshold']) and DEFAULT_TRAINING_THRESHOLD != project.command_line_arguments['training_threshold']:
                    print("Property satisfied!")
                    satisfied = True

            print(episode, "Episode\tReward", episode_reward, '\tAverage Reward', reward_of_sliding_window, '\tBest Average Reward', best_reward_of_sliding_window)
            gc.collect()
            if satisfied:
                break
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        gc.collect()
    finally:
        torch.cuda.empty_cache()

    return best_reward_of_sliding_window