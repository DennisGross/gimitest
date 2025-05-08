# Gimitest
The _Gimitest_ software enhances the [Farama Gymnasium](https://gymnasium.farama.org/index.html), [PettingZoo](https://pettingzoo.farama.org/content/basic_usage/), and similar frameworks by modifying its `reset(...)` and `step(...)` methods, thereby simplifying the testing process of Reinforcement Learning (RL) agents at specific time steps and episode terminations, whether in training or testing phases.

A decorator design pattern is used to modify the functionality of an object at runtime.
We use the decorator design pattern to extend the `reset(...)` and `step(...)` methods of the single-agent and multi-agent RL environments, allowing for testing at specific time intervals and episode terminations.

**Customizable Gimitest capabilities** for single-agent and multi-agent systems include:
- *Search-based testing*
- *Adversarial testing*
- *Metamorphic testing*
- *Logging capabilities*
- *Automated testing*

See examples in the `examples` folder.
Watch the [video](https://youtu.be/9WiWZyrUhLw) for a quick demonstration.


## 🚀 Getting Started
Install package via:

```
pip install git+https://github.com/DennisGross/gimitest.git
```

Gimitest allows us to decorate our first environment with **only a few extra lines**:
```
import gymnasium as gym
from gimitest.env_decorator import EnvDecorator
from gimitest.gtest import GTest
# Create environment
env = gym.make('CartPole-v1')
# Create Gimitest object
m_gtest = GTest(env)
# Decorate the environment with it
EnvDecorator.decorate(env, m_gtest)
# THE REST IS THE SAME...
```

## 👮🏼‍♂️ GTest
While `GTest` serves as a base class, it's designed to be flexible and extendable. Users can create custom GTest subclasses and override specific methods to suit their testing needs.

The `step(...)`-wrapping first executes the `pre_step_configuration(...)` method, then the `pre_step_test(...)` method, then the original `step(...)` method, then the `post_step_test(...)` method, and finally the `post_step_configuration(...)` method.

The `reset(...)`-wrapping first executes the `pre_reset_test(...)` method, then the `pre_reset_configuration(...)` method, then the original `reset(...)` method, then the `post_reset_test(...)` method, and finally the `post_reset_configuration(...)` method.

### Configuration Methods
The configuration methods allow us to override the way how your `GTest` can modify the environment. For instance, changing the gravity or cart mass in the ccartpole environment at different times (pre/post/reset/step):
- `pre_step_configuration(...)`
- `post_step_configuration(...)`
- `pre_reset_configuration(...)`
- `post_reset_configuration(...)`

To access internal environment parameters, we can use the `original_env = env.unwrapped` to unwrap the environments and access the attributes as usually (`original_env.ATTRIBUTE_NAME`).
However, sometims this does not work and Gimitest allows via `get_attribute(...)` and `set_attribute(...)` to modify internal environment parameters, too.


### Testing Methods
The testing methods allow us to create specific tests at different times (pre/post/reset/step):
- `pre_step_test(...)`
- `post_step_test(...)`
- `pre_reset_test(...)`
- `post_reset_test(...)`


## 📊 GLogger
`GLogger` allows us to log the whole testing process at every point in time.
Just decorate `GTest` with a `GLogger`:
```
from gimitest.glogger import GLogger
m_logger = GLogger("m_log")
GTestDecorator.decorate_with_logger(m_gtest, m_logger)
```

## 🛠️ Modifications
If you want to modify Gimitest, please follow the steps below:

1. Clone the repository:
```
git clone https://github.com/DennisGross/gimitest.git
```

2. Install the requirements:
```
pip install -r requirements.txt
```

3. Modify the code as you wish.



## 📜 Citation
If you use Gimitest in your research, please cite the following:
```
@software{Gross_gimitest_2025,
  author = {Gross, Dennis and Mazouni, Quentin and Spieker, Helge},
  license = {MIT},
  month = may,
  title = {{Gimitest: A Comprehensive Tool for Testing Reinforcement Learning Policies}},
  url = {https://github.com/DennisGross/gimitest},
  version = {1.0},
  year = {2025}
}
```