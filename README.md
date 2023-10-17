# Gimitest
The _Gimitest framework_ enhances the [Farama Gymnasium](https://gymnasium.farama.org/index.html) by modifying its `reset()` and `step()` methods, thereby simplifying the testing process of Reinforcement Learning (RL) agents at specific time steps and episode terminations.
It offers predefined `TestCases` along with the capability to develop custom `TestCases`, thereby providing flexibility in the testing regime. The availability of both standard and customizable `Configurators` further enables the sampling of RL agent behavior under varied initial states and environment parameters.

## Setup
Install package via:
`pip install git+https://github.com/DennisGross/gimitest.git`


## Getting Started
The example code snippet demonstrates how to set up a basic environment, using the 'CartPole-v1' environment as an example.
First Initialize the Gym Environment: Utilize the gym.make() function from the Gymnasium package to create your environment.
Second, create a list of test cases using the `TestCase` class to define the conditions under which your RL agent will be evaluated.
Third, utilize the `Configurator` class to set initial conditions, such as state variables.
Fourth, the state variable needs to be specified via the parameters and needs to be available as attribute in the environment. In our case, the state is stored in the state variable `state`.
Fifth, the GymDecorator class is employed to extend the `reset()` and `step()` methods of the Gym environment, allowing for testing at specific time intervals and episode terminations.
Then, the decorated environment can then be executed to evaluate the RL agent's performance.
```
import gymnasium as gym
from gym_decorator import GymDecorator
from test_cases.test_case import TestCase
from configurators.configurator import Configurator

# Init Gym
env = gym.make('CartPole-v1')
# List of Test Cases
test_cases = [TestCase()]
# Configurator
configurator = Configurator({"state_variable_name": "state"})
# Decorate the environment by extending its reset and step function
env = GymDecorator.decorate_gym(env, test_cases, configurator)

# Run the environment
state, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    if done or truncated:
        state, info = env.reset()
```