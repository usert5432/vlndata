from abc import ABC, abstractmethod
import numpy as np

from vlndata.funcs import Spec, unpack_name_args

class Noise(ABC):

    def __init__(self, seed):
        self._prg = np.random.default_rng(seed)

    @abstractmethod
    def generate(self, shape):
        raise NotImplementedError

class DebugNoise(Noise):

    def __init__(self, value, seed = 0):
        super().__init__(seed)
        self._value = value

    def generate(self, shape) -> np.ndarray:
        return np.full(shape, self._value)

class UniformNoise(Noise):

    def __init__(self, a, b, seed = 0):
        super().__init__(seed)
        self._a = a
        self._b = b

    def generate(self, shape):
        return self._prg.uniform(self._a, self._b, shape)

class DiscreteNoise(Noise):

    def __init__(self, values, prob = None, seed = 0):
        super().__init__(seed)
        self._values = values
        self._prob   = prob

    def generate(self, shape):
        return self._prg.choice(
            self._values, size = shape, replace = True, p = self._prob
        )

class GaussianNoise(Noise):

    def __init__(self, mu, sigma, seed = 0):
        super().__init__(seed)
        self._mu    = mu
        self._sigma = sigma

    def generate(self, shape):
        return self._prg.normal(self._mu, self._sigma, shape)

def select_noise(noise : Spec) -> Noise:
    name, kwargs = unpack_name_args(noise)

    if name in [ 'gaussian', 'normal' ]:
        return GaussianNoise(**kwargs)

    if name == 'discrete':
        return DiscreteNoise(**kwargs)

    if name == 'uniform':
        return UniformNoise(**kwargs)

    if name == 'debug':
        return DebugNoise(**kwargs)

    raise ValueError("Unknown noise: %s" % name)

