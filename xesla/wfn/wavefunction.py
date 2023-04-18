#!/usr/bin/python3

from abc import ABCMeta, abstractmethod, abstractproperty
from xesla.opt.function import Function

class Wavefunction(Function,metaclass=ABCMeta):
    """Abstract base class for a wavefunction object"""

    @abstractproperty
    def energy(self):
        """Get the wavefunction energy"""
        pass

    @abstractproperty
    def gradient(self):
        """Get the function gradient"""
        pass

    @abstractproperty
    def hessian(self):
        """Get the Hessian matrix of second-derivatives"""
        pass

    @abstractmethod
    def take_step(self,step):
        """Take a step in parameter space"""
        pass

    @abstractmethod
    def save_last_step(self):
        """Save current position"""
        pass

    @abstractmethod
    def restore_last_step(self):
        """Return wavefunction to previous position"""
        pass

    @property
    def value(self):
        """Map the energy onto the function value"""
        return self.energy
