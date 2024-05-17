#!/usr/bin/python3

from abc import ABCMeta, abstractmethod, abstractproperty
from exelsis.opt.function import Function

class Wavefunction(Function,metaclass=ABCMeta):
    """Abstract base class for a wavefunction object"""
    @property 
    @abstractmethod
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

    @abstractmethod
    def save_to_disk(self, tag):
        """Save a wavefunction object to disk"""
        pass

    @abstractmethod
    def read_from_disk(self, tag):
        """Read a wavefunction object to disk"""
        pass

    @abstractmethod 
    def hamiltonian(self, other):
        """Compute the Hamiltonian coupling with another wavefunction of this type"""
        pass
