#!/usr/bin/python3

from abc import ABCMeta, abstractmethod, abstractproperty
from xesla.opt.function import Function

class Wavefunction(Function,metaclass=ABCMeta):
    """Abstract base class for a wavefunction object"""
    @abstractproperty
    def energy(self):
        """Get the wavefunction energy"""
        pass

    @property
    def value(self):
        """Map the energy onto the function value"""
        return self.energy

    @abstractmethod 
    def overlap(self,other):
        """Compute the overlap with another wavefunction of this type"""
        pass

    @abstractmethod
    def initialise(self, mo_coeff, mat_ci=None, integrals=True):
        """Initialise wavefunction with orbital coefficient (and CI matrix)"""
        pass
