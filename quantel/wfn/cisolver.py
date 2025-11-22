#!/usr/bin/python3
# Author: Hugh G. A. Burton

import numpy as np
import quantel
from quantel.opt.davidson import Davidson

class ArbitraryCI:
    """
    Class for solving the arbitrary CI problem using the Davidson algorithm.
    """
    def __init__(self, cispace):
        """
        Initialise the ArbitraryCI instance from cispace object.
        """
        # Save the cispace object
        self.cispace = cispace
        # Function to perform Hamiltonian action on a vector
        self.H_on_vec = lambda v : self.cispace.H_on_vec(v)
        # Diagonal of the Hamiltonian
        self.Hd = self.cispace.build_Hd()

    @property
    def ndet(self):
        """ Number of determinants in the CI space."""
        return self.cispace.ndet()
    
    @property
    def nelec(self):
        """ Number of electrons in the CI space."""
        return (self.cispace.nalfa(), self.cispace.nbeta())

    def solve(self, nroots, xguess=None, maxit=100, tol=1e-6, verbose=1):
        """
        Solve the arbitrary CI problem using the Davidson algorithm.
        """
        # Set initial guess to lowest identity vectors if requested
        if(xguess is None):
            xguess = np.eye(self.ndet, nroots)

        # Run the Davidson algorithm
        davidson = Davidson()
        self.eigval, self.eigvec = davidson.run(self.H_on_vec, self.Hd, nroots, 
                                                xguess=xguess, maxit=maxit, tol=tol, plev=verbose)
        return self.eigval, self.eigvec

    def get_det_index(self, occstr):
        return self.cispace.get_det_index(quantel.Determinant(occstr))

    def write_cidump(self,tag,civec=None):
        # Write the CI vector dump
        from quantel.utils.ci_utils import write_cidump
        if civec is None:
            vec = list(zip(self.cispace.get_det_list(),self.eigvec[:,0]))
        else:
            vec = list(zip(self.cispace.get_det_list(),civec))
        write_cidump(vec,self.ncore,self.nmo,tag+'_civec.txt')

    def get_hamiltonian(self):
        """
        Get the Hamiltonian matrix of the CI problem.
        """
        return self.cispace.build_Hmat()

class FCI(ArbitraryCI):
    """
    Class for solving the full CI problem using the Davidson algorithm.
    """
    def __init__(self, mo_ints, nelec):
        """
        Initialise the FCI instance from cispace object.
        """
        # Save mo_ints object
        self.mo_ints = mo_ints
        # Number of core orbitals
        self.ncore = mo_ints.ncore()
        # Number of correlated orbitals
        self.nact = mo_ints.nact()
        # Number of molecular orbitals
        self.nmo = mo_ints.nmo()
        # Number of electrons
        self.nalfa = nelec[0]
        self.nbeta = nelec[1]

        # Check the input
        if(self.nalfa < 0):
            raise ValueError("Number of electrons cannot be negative.")
        if(self.nbeta < 0):
            raise ValueError("Number of electrons cannot be negative.")

        # Create the CI space object
        cispace = quantel.CIspace(mo_ints,self.nact,self.nalfa,self.nbeta)
        cispace.initialize('FCI')

        # Call the parent constructor
        super().__init__(cispace)

class CASCI(ArbitraryCI):
    """
    Class for solving the CAS CI problem using the Davidson algorithm.
    """
    def __init__(self, mo_ints, nelec, ncas):
        """
        Initialise a CASCI instance from cispace object.
        """
        # Save mo_ints object
        self.mo_ints = mo_ints
        # Number of core orbitals
        self.ncore = mo_ints.ncore()
        # Number of correlated orbitals
        self.nact = ncas
        # Number of molecular orbitals
        self.nmo = mo_ints.nmo()
        # Number of electrons
        self.nalfa = nelec[0]
        self.nbeta = nelec[1]

        # Check the input
        if(self.nalfa < 0):
            raise ValueError("Number of electrons cannot be negative.")
        if(self.nbeta < 0):
            raise ValueError("Number of electrons cannot be negative.")
        if(self.nalfa + self.nbeta > 2*self.nact):
            raise ValueError("Number of electrons exceeds number of active orbitals.")

        # Create the CI space object
        cispace = quantel.CIspace(mo_ints,self.nact,self.nalfa,self.nbeta)
        cispace.initialize('CASCI')

        # Call the parent constructor
        super().__init__(cispace)