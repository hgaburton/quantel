#!/usr/bin/python3
# Author: Hugh G. A. Burton

import numpy as np
import quantel
from quantel.opt.davidson import Davidson
from quantel.ints.pyscf_integrals import PySCF_MO_Integrals

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
        #self.res_Hd = self.cispace.resolve_build_Hd()

    @property
    def nconfigs(self):
        """ Number of determinants in the CI space."""
        return self.cispace.nconfigs()
    
    def solve(self, nroots, xguess=None, maxit=100, tol=1e-6, verbose=1):
        """
        Solve the arbitrary CI problem using the Davidson algorithm.
        """
        # Set initial guess to lowest identity vectors if requested
        if(xguess is None):
            xguess = np.eye(self.nconfigs, nroots) + np.random.rand(self.nconfigs, nroots)*0.1
        # Run the Davidson algorithm
        davidson = Davidson()
        self.eigval, self.eigvec = davidson.run(self.H_on_vec, self.Hd, nroots, 
                                                xguess=xguess, maxit=maxit, tol=tol, plev=verbose)
        return self.eigval, self.eigvec

    def get_config_index(self, conf_vec):
        return self.cispace.get_det_index(quantel.Configuration(conf_vec))
    
    def get_config_list(self):
        return self.cispace.get_config_list()

    def write_cidump(self,tag,nmo,civec=None,ncore=0):
        # Write the CI vector dump
        from quantel.utils.ci_utils import write_cidump
        if civec is None:
            vec = list(zip(self.cispace.get_config_list(),self.eigvec[:,0]))
        else:
            vec = list(zip(self.cispace.get_config_list(),civec))
        write_cidump(vec,ncore,nmo,tag+'_civec.txt')

    def get_hamiltonian(self):
        """
        Get the Hamiltonian matrix of the CI problem.
        """
        return self.cispace.build_Hmat()
    
    def get_s2(self,vec):
        """
        Compute the <S^2> expectation value for a given CI vector.
        """
        return self.totspin*(self.totspin+1)  

class FCI(ArbitraryCI):
    """
    Class for solving the full CI problem using the Davidson algorithm.
    """
    def __init__(self, mo_ints, nelec, totspin, version=1):
        """
        Initialise the FCI instance from cispace object.
        """
        # Convert integral object if needed
        if isinstance(mo_ints, PySCF_MO_Integrals):
            self.mo_ints = mo_ints.get_quantel_ints()
        else:
            self.mo_ints = mo_ints
        # Number of correlated orbitals
        self.nmo = mo_ints.nmo()
        self.nelec = nelec
        self.totspin = totspin
 
        # Check the input
        if(self.nelec < 0):
            raise ValueError("Number of electrons cannot be negative.")
        if(self.nelec > 2*self.nmo):
            raise ValueError("Number of electrons exceeds number of active orbitals.")

        # Create the CI space object
        cispace = quantel.GUGA_CIspace(self.mo_ints,self.nmo,self.nelec, self.totspin)
        cispace.initialize('FCI')

        # Call the parent constructor
        super().__init__(cispace)


