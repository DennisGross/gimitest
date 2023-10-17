# Gimitest
The _Gimitest framework_ enhances the [Farama Gymnasium](https://gymnasium.farama.org/index.html) by modifying its `reset(...)` and `step(...)` methods, thereby simplifying the testing process of Reinforcement Learning (RL) agents at specific time steps and episode terminations, whether in training or testing phases.
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
Fifth, the GymDecorator class is employed to extend the `reset(...)` and `step(...)` methods of the Gym environment, allowing for testing at specific time intervals and episode terminations.
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

## Architecture
Decorator design pattern is used to modify the functionality of an object at runtime. At the same time other instances of the same class will not be affected by this, so individual object gets the modified behavior.

We use the decorator design pattern to extend the `reset(...)` and `step(...)` methods of the Gym environment, allowing for testing at specific time intervals and episode terminations.

In the case of `reset(...)`, the decorator checks the current state of the environment and compares it to the test cases. It then executes the basic functionality of the `reset(...)` method.
After that, it executes the configurator to either modify the initial state or modify any other parameter (such as the gravity of the environment) and returns the modified state with the corresponding information.
The TestCases can inform the configurator via messages about their execution and guide the configuration.
The following code snippet shows the implementation of the `reset(...)` decorator.
It first executes the test cases and stores their messages.
Then, it calls the original `reset(...)` function and stores the next state.
Finally, it applies the configurator if set and returns the modified state with the corresponding information.
```
def wrapper(*args, **kwargs):
    test_case_messages = []
    # Handle test cases if any
    if test_cases is not None:
    for test_case in test_cases:
        test_case.episode_execute()
        test_case_messages.append(test_case.get_message())
        test_case.episode_store()

    # Call the original reset function
    next_state, info = original_reset_function(*args, **kwargs)
    env.tmp_storage_of_state = next_state

    # Apply configurator if set
    if configurator is not None:
        env.tmp_storage_of_state = configurator.configure(env, test_case_messages)
    return env.tmp_storage_of_state, info
```

In the case of `step(...)`, the decorator checks the current state of the environment and compares it to the test cases. It then executes the basic functionality of the `step(...)` method.

### TestCase-Class
The `TestCase` class serves as a base class for creating test cases specifically tailored for gym environments. It contains a single attribute, parameters, which is a dictionary meant for holding custom parameters for individual test cases. The class has various methods that can be overridden to provide custom behavior during testing. The `__init__(...)` method initializes the class instance with these custom parameters. The `step_execute(...)` method is designed to be called at each step in the gym environment, taking various arguments like current state, action arguments, and original outcomes like next state and reward. It returns potentially modified versions of these outcomes. The `step_store(...)` and `step_load(...)` methods are placeholders for storing and loading data relevant to each step, respectively. Similarly, `episode_execute(...)`, `episode_store(...)`, and episode_load methods serve as placeholders for operations at the start and end of each episode. Lastly, the `get_message method(...)` is designed to return messages or information as a dictionary to inform the configurator about the execution of the test case.

### Configurator-Class
The `Configurator` class is a foundational class intended to configure gym environments.
It has a single attribute, parameters, a dictionary expected to contain custom configuration parameters including a key for "state_variable_name" which indicates the name of the state variable in the gym environment. The `__init__(...)` method initializes the object with given parameters, expected to contain a key for "state_variable_name".
The `modify_state(...)` method alters the state of the gym environment based on the parameters.
the `get_state(...)` method etrieves the current state of the gym environment.
The `configure(...)` method is intended for overriding to provide custom environment configuration logic.