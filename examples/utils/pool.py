import numpy as np


from abc import ABC, abstractmethod
from typing import List, Tuple


class Pool(ABC):
    '''Compared to the paper, the pool features the storage of crashes, which correspond to failure-triggering states/inputs.'''
    def __init__(self) -> None:
        super().__init__()
        self.inputs = [] # type: List[np.ndarray]
        self.rewards = [] # type: List[float]
        self.oracles = [] # type: List[int]
        self.sensitivities = [] # type: List[float]
        self.coverages = [] # type: List[float]
        self.selected = [] # type: List[int]
        self.crashes = [] # type: List[np.ndarray]

        self.delimiter = ' '


    @abstractmethod
    def add(self, input: np.ndarray, acc_reward: float, coverage: float, sensitivity: float, oracle: bool) -> None:
        pass

    def add_crash(self, state: np.ndarray):
        '''Keeps track of failure-triggering states by adding @state to the list of crashes.'''
        self.crashes.append(state.copy())

    def select(self, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
        '''Returns one of the inputs along with its accumulated reward of the pool with random sampling biased by the sensitivities.'''
        if np.sum(self.sensitivities) == 0:
            index = rng.choice(len(self.inputs))
        else:
            index = rng.choice(len(self.inputs), p=(self.sensitivities / np.sum(self.sensitivities)))
        self.selected[index] += 1
        # copy.deepcopy(self.inputs[index]) # .copy(() seems to be enough
        return self.inputs[index].copy(), self.rewards[index]

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


class IndexedPool(Pool):
    '''
    This implementation uses string representations of the inputs as keys to avoid adding redundant inputs.
    It is thus suited for small input space and detailed results.
    '''
    def __init__(self, is_integer: bool = False) -> None:
        super().__init__()
        self.indices = [] # type: List[str]
        self.added = [] # type: List[int]

        self.is_integer = is_integer


    def _key(self, input: np.ndarray) -> str:
        '''Returns the "key" format of an input.'''
        if self.is_integer:
            return self.delimiter.join(['{:.0f}'.format(i) for i in input])
        else:
            return self.delimiter.join([str(i) for i in input])


    def add(self, input: np.ndarray, acc_reward: float, coverage: float, sensitivity: float, oracle: bool):
        '''
        Adds a test case result to the pool.
        If the input/state has already been evaluated, the results are erased.
        '''
        key = self._key(input)
        try:
            index = self.indices.index(key)
        except ValueError:
            index = None

        if index is None:
            self.indices.append(key)
            self.inputs.append(input)
            self.rewards.append(acc_reward)
            self.coverages.append(coverage)
            self.sensitivities.append(sensitivity)
            self.added.append(0)
            self.selected.append(0)
            self.oracles.append(int(oracle))
        else:
            # print(f'input {key} is already in the pool!')
            self.rewards[index] = acc_reward
            self.coverages[index] = coverage
            self.sensitivities[index] = sensitivity
            self.added[index] += 1
            self.oracles[index] = int(oracle)
            # does nothing for selection tracking


    def save(self, path: str):
        '''Saves all the lists of the pool instance.'''
        # if Python 3.8 and higher
        # x = 5
        # variable_name = f"{x=}".split("=")[0]
        # print(variable_name)
        if self.is_integer:
            np.savetxt(path + '_inputs.txt', self.inputs, fmt='%1.0f', delimiter=',')
            np.savetxt(path + '_crashes.txt', self.crashes, fmt='%1.0f', delimiter=',')
        else:
            np.savetxt(path + '_inputs.txt', self.inputs, delimiter=',')
            np.savetxt(path + '_crashes.txt', self.crashes, delimiter=',')

        np.savetxt(path + '_added.txt', self.added, fmt='%1.0f', delimiter=',')
        np.savetxt(path + '_selected.txt', self.selected, fmt='%1.0f', delimiter=',')
        np.savetxt(path + '_oracles.txt', self.oracles, fmt='%1.0f', delimiter=',')
        np.savetxt(path + '_rewards.txt', self.rewards, delimiter=',')
        np.savetxt(path + '_sensitivities.txt', self.sensitivities, delimiter=',')
        np.savetxt(path + '_coverages.txt', self.coverages, delimiter=',')


    #TODO: loading integer values does not work (but it is not an used feature)
    def load(self, path: str, as_integers: bool = False):
        '''Loads the results of a pool instance by loading all the lists.'''
        self.is_integer = as_integers
        if self.is_integer:
            self.is_integer = True
            self.inputs = [i for i in np.loadtxt(path + '_inputs.txt', delimiter=',', dtype=int)]
            self.crashes = [i for i in np.loadtxt(path + '_crashes.txt', delimiter=',', dtype=int)]
        else:
            self.inputs = [i for i in np.loadtxt(path + '_inputs.txt', delimiter=',', dtype=float)]
            self.crashes = [i for i in np.loadtxt(path + '_crashes.txt', delimiter=',', dtype=float)]

        self.indices = [self._key(i) for i in self.inputs]
        self.added = [i for i in np.loadtxt(path + '_added.txt', delimiter=',', dtype=int)]
        self.selected = [i for i in np.loadtxt(path + '_selected.txt', delimiter=',', dtype=int)]
        self.oracles = [i for i in np.loadtxt(path + '_oracles.txt', delimiter=',', dtype=int)]
        self.rewards = [i for i in np.loadtxt(path + '_rewards.txt', delimiter=',', dtype=float)]
        self.sensitivities = [i for i in np.loadtxt(path + '_sensitivities.txt', delimiter=',', dtype=float)]
        self.coverages = [i for i in np.loadtxt(path + '_coverages.txt', delimiter=',', dtype=float)]

        tmp = len(self.inputs)
        assert len(self.added) == tmp
        assert len(self.selected) == tmp
        assert len(self.indices) == tmp
        assert len(self.rewards) == tmp
        assert len(self.sensitivities) == tmp
        assert len(self.coverages) == tmp
        assert len(self.oracles) == tmp
