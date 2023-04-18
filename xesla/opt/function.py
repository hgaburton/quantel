#!/usr/bin/python3

from abc import ABCMeta, abstractmethod, abstractproperty

class Function(metaclass=ABCMeta):
    """Abstract base class for a real-valued objective function"""

    @abstractproperty
    def value(self):
        """Get the corresponding variational value"""
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
