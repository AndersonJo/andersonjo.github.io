from abc import ABCMeta, abstractmethod
import numpy as np


class ActivationFunction(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def activate(self, x):
        pass

    @abstractmethod
    def derivative(self, d):
        pass


class SigmoidActivationFunction(ActivationFunction):
    def activate(self, x):
        return 1. / (1. + np.e**(-1.*x))

    def derivative(self, d):
        return d * (1. - d)
