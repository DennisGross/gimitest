# Gimitest
The _Gimitest framework_ enhances the [Farama Gymnasium](https://gymnasium.farama.org/index.html) and [PettingZoo](https://pettingzoo.farama.org/content/basic_usage/) by modifying its `reset(...)` and `step(...)` methods, thereby simplifying the testing process of Reinforcement Learning (RL) agents at specific time steps and episode terminations, whether in training or testing phases.

A decorator design pattern is used to modify the functionality of an object at runtime.
We use the decorator design pattern to extend the `reset(...)` and `step(...)` methods of the single-agent and multi-agent RL environments, allowing for testing at specific time intervals and episode terminations.

## 🚀 Getting Started
Install package via:
`pip install git+https://github.com/DennisGross/gimitest.git`
Gimitest allows us to decorate our first environment with **only two extra lines**:
```
# Create environment
env = gym.make('CartPole-v1')
# Create Gimitest object
m_gtest = GTest()
# Decorate the environment with it
EnvDecorator.decorate(env, m_gtest)
# THE REST IS THE SAME...
```

## 👮🏼‍♂️ GTest
Of course the `GTest` is only a base class.
However, we can just create your own `GTest` by inheriting the class and overwrite the following methods.


#### Configuration Methods
The configuration methods allow us to override the way how your `GTest` can modify the environment. For instance, changing the gravity or cart mass in the ccartpole environment at different times (pre/post/reset/step):
- `pre_step_configuration(...)`
- `post_step_configuration(...)`
- `pre_reset_configuration(...)`
- `post_reset_configuration(...)`

To access internal environment parameters, we can use the `original_env = env.unwrapped` to unwrap the environments and access the attributes as usually (`original_env.ATTRIBUTE_NAME`).
However, sometims this does not work and Gimitest allows via `get_attribute(...)` and `set_attribute(...)` to modify internal environment parameters, too.


#### Testing Methods
The testing methods allow us to create specific tests at different times (pre/post/reset/step):
- `pre_step_test(...)`
- `post_step_test(...)`
- `pre_reset_test(...)`
- `post_reset_test(...)`

### 📊 Additional Functionality
`GLogger` allows us to log the whole testing process at every point in time.
Just decorate `GTest` with a `GLogger`:
```
m_logger = GLogger("m_log")
GTestDecorator.decorate_with_logger(m_gtest, m_logger)
```
`GAnalyse` gives us further functionality to analyse the logs.
To analyse the logs use the logger:
```
g_analyse = GAnalyse(m_logger)
# DO YOUR STUFF
```