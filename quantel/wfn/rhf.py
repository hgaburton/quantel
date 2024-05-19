#!/usr/bin/python3
# Author: Hugh G. A. Burton

import numpy as np
import scipy.linalg
from quantel.utils.linalg import delta_kron, orthogonalise
from .wavefunction import Wavefunction

class RHF(Wavefunction):
    """ Restricted Hartree-Fock method

        Inherits from the Wavefunction abstract base class with pure virtual properties:
            - energy
            - gradient
            - hessian 
            - take_step
            - save_last_step
            - restore_step
    """
    def __init__(self, integrals, verbose=0):
        """Initialise Restricted Hartree-Fock wave function
               integrals : quantel integral interface
               verbose   : verbosity level
        """
        self.integrals = integrals
        self.nalfa     = integrals.molecule().nalfa()
        self.nbeta     = integrals.molecule().nbeta()

        # Get number of basis functions and linearly independent orbitals
        self.norb      = integrals.nbsf()
        self.nmo       = integrals.nmo()

        # For now, we assume that the number of alpha and beta electrons are the same
        assert(self.nalfa == self.nbeta)
        self.nocc      = self.nalfa
        self.verbose   = verbose

        # Setup the indices for relevant orbital rotations
        self.rot_idx   = self.uniq_var_indices() # Indices for orbital rotations
        self.nrot      = np.sum(self.rot_idx) # Number of orbital rotations

    @property
    def dim(self):
        """Get the number of degrees of freedom"""
        return self.nrot

    @property
    def energy(self):
        """Get the energy of the current RHF state"""
        # Nuclear potential
        E  = self.integrals.scalar_potential()
        E += np.einsum('pq,pq', self.integrals.oei_matrix(True)+self.fock, self.dens, optimize="optimal")
        return E

    @property
    def s2(self):
        ''' Compute the spin of a given FCI vector '''
        # TODO Need to implement s2 computation
        return 0 #self.fcisolver.spin_square(self.mat_ci[:,0], self.ncas, self.nelecas)[0]

    @property
    def gradient(self):
        """Get the energy gradient with respect to the orbital rotations"""
        g = 4 * np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])
        return g[self.rot_idx]

    @property
    def hessian(self):
        ''' This method concatenate the orb-orb, orb-CI and CI-CI part of the Hessian '''
        return 0
    

    def save_to_disk(self,tag):
        """Save object to disk with prefix 'tag'"""
        # Get the Hessian index
        #hindices = self.get_hessian_index()

        # Save coefficients, CI, and energy
        #np.savetxt(tag+'.mo_coeff', self.mo_coeff, fmt="% 20.16f")
        #np.savetxt(tag+'.mat_ci',   self.mat_ci, fmt="% 20.16f")
        #np.savetxt(tag+'.energy',   
        #           np.array([[self.energy, hindices[0], hindices[1], self.s2]]), 
        #           fmt="% 18.12f % 5d % 5d % 12.6f")


    def read_from_disk(self,tag):
        """Read object from disk with prefix 'tag'"""
        # Read MO coefficient and CI coefficients
        #mo_coeff = np.genfromtxt(tag+".mo_coeff")
        #ci_coeff = np.genfromtxt(tag+".mat_ci")

        # Initialise object
        #self.initialise(mo_coeff, ci_coeff)

    def copy(self):
        # Return a copy of the current object
        #newcas = ESMF(self.mol, spin=self.spin, ref_allowed=self.with_ref)
        #newcas.initialise(self.mo_coeff, self.mat_ci, integrals=False)
        return newcas

    def overlap(self, them):
        """Compute the many-body overlap with another CAS waveunction (them)"""
        return 0 #esmf_coupling(self, them, self.ovlp, with_ref=self.with_ref)[0]

    def hamiltonian(self, them):
        """Compute the many-body Hamiltonian coupling with another CAS wavefunction (them)"""
        #eri = ao2mo.restore(1, self._scf._eri, self.mol.nao).reshape(self.mol.nao**2, self.mol.nao**2)
        return 0 #esmf_coupling(self, them, self.ovlp, self.hcore, eri, self.enuc, with_ref=self.with_ref)

    def initialise(self, mo_guess):
        """Initialise the wave function with a set of molecular orbital coefficients"""
        self.mo_coeff = orthogonalise(mo_guess, self.integrals.overlap_matrix())
        self.update()

    def update(self):
        """Update the 1RDM and Fock matrix for the current state"""
        self.get_density()
        self.get_fock()

    def get_density(self):
        """Compute the 1RDM for the current state"""
        Cocc = self.mo_coeff[:,:self.nocc]
        self.dens = np.dot(Cocc, Cocc.T)

    def get_fock(self):
        """Compute the Fock matrix for the current state"""
        self.fock = self.integrals.build_fock(self.dens)

    def restore_last_step(self):
        """Restore orbital coefficients to the previous step"""
        self.mo_coeff = self.mo_coeff_save.copy()

    def save_last_step(self):
        """Save the current orbital coefficients"""
        self.mo_coeff_save = self.mo_coeff.copy()

    def take_step(self,step):
        """Take a step in the orbital space"""
        self.save_last_step()
        self.rotate_orb(step[:self.nrot])

    def rotate_orb(self,step): 
        """Rotate molecular orbital coefficients with a step"""
        # Build the anti-symmetric step matrix
        K = np.zeros((self.nmo,self.nmo))
        K[self.rot_idx] = step

        # Build the unitary transformation
        Q = scipy.linalg.expm(K - K.T)
        
        # Transform the coefficients
        self.mo_coeff = np.dot(self.mo_coeff, Q)

        # Update the density and fock matrices
        self.update()

    def uniq_var_indices(self):
        """Create a matrix of boolean of size (norb,norb). 
           A True element means that this rotation should be taken into account during the optimization.
        """
        # Include only occupied-virtual rotations
        mask = np.zeros((self.nmo,self.nmo), dtype=bool)
        mask[self.nocc:,:self.nocc] = True
        return mask