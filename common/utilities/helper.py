"""This module provides helper functions for COOL-MC."""
import argparse
import sys
import random
from typing import Any, Dict
import numpy as np
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAINING_TASK = "training"
DEFAULT_TRAINING_THRESHOLD = -1000000000000

def get_arguments() -> Dict[str, Any]:
    """Parses all the command line arguments
    Returns:
        Dict[str, Any]: dictionary with the command line arguments as key and their assignment as value
    """
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()

    # Meta
    arg_parser.add_argument('--task', help=f'What type of task do you want to perform({TRAINING_TASK}, rl_model_checking)?', type=str,
                            default=TRAINING_TASK)
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='defaultproject')
    arg_parser.add_argument('--env', help='Name of the OpenAI Gym environment (CartPole-v1, MountainCar-v0, Acrobot-v1, FrozenLake-v1).', type=str,
                            default='CartPole-v1')
    arg_parser.add_argument('--state_attribute', help='Name of the state attribute to be modified.', type=str,
                            default='state')
    arg_parser.add_argument('--seed', help='Random Seed for numpy, random, storm, pytorch', type=int,
                            default=-1)
    arg_parser.add_argument('--training_threshold', help='Range Plotting Flag.', type=float,
                            default=DEFAULT_TRAINING_THRESHOLD)
    # Training
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=101)
    arg_parser.add_argument('--eval_interval', help='Monitor every eval_interval episodes.', type=int,
                            default=9)
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment.', type=int,
                            default=100)
    arg_parser.add_argument('--deploy', help='Deploy Flag (0=no deploy, 1=deploy).', type=int,
                            default=0)

    # Preprocessor
    arg_parser.add_argument('--preprocessor', help='Preprocessor configuration string.', type=str,
                            default='')


    # Agents
    arg_parser.add_argument('--rl_algorithm', help='What is the used RL algorithm?', type=str,
                            default='dqn_agent')
    arg_parser.add_argument('--layers', help='Number of layers', type=int,
                            default=2)
    arg_parser.add_argument('--neurons', help='Number of neurons per layer', type=int,
                            default=64)
    arg_parser.add_argument('--replay_buffer_size', help='Replay buffer size', type=int,
                            default=300000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.9999)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=304)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.0001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=32)

    args, _ = arg_parser.parse_known_args(sys.argv)
    return vars(args)



def set_random_seed(seed: int):
    """Set global seed to all used libraries. If you use other libraries too,
    add them here.
    Args:
        seed (int): Random Seed
    """
    assert isinstance(seed, int)
    if seed != -1:
        print("Set Seed to", seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

