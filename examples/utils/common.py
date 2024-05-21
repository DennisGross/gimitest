from stable_baselines3 import PPO, DQN


def load_dqn(model_path: str):
    return DQN.load(model_path, device='cpu')


def load_ppo(model_path: str, device = 'cpu'):
    return PPO.load(model_path, device=device)