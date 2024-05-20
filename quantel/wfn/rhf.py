#!/usr/bin/python3
# Author: Hugh G. A. Burton

import numpy as np
import scipy.linalg
from quantel.utils.linalg import orthogonalise
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
        self.nbsf      = integrals.nbsf()
        self.nmo       = integrals.nmo()

        # For now, we assume that the number of alpha and beta electrons are the same
        assert(self.nalfa == self.nbeta)
        self.nocc      = self.nalfa
        self.verbose   = verbose

        # Setup the indices for relevant orbital rotations
        self.rot_idx   = self.uniq_var_indices() # Indices for orbital rotations
        self.nrot      = np.sum(self.rot_idx) # Number of orbital rotations

        # Define the orbital energies and coefficients
        self.mo_coeff         = None
        self.orbital_energies = None

    @property
    def dim(self):
        """Get the number of degrees of freedom"""
        return self.nrot

    @property
    def energy(self):
        """Get the energy of the current RHF state"""
        # Nuclear potential
        E  = self.integrals.scalar_potential()
        E += np.einsum('pq,pq', self.integrals.oei_matrix(True) + self.fock, self.dens, optimize="optimal")
        return E

    @property
    def s2(self):
        """Get the spin of the current RHF state"""
        return 0 # All RHF states have spin 0

    @property
    def gradient(self):
        """Get the energy gradient with respect to the orbital rotations"""
        g = 4 * np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])
        return g[self.rot_idx]

    @property
    def hessian(self):
        """Compute the internal RHF orbital Hessian"""
        # Number of occupied and virtual orbitals
        no = self.nocc
        nv = self.nmo - self.nocc
        # Compute Fock matrix in MO basis 
        Fmo = np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])
        # Compute ao_to_mo integral transform
        eri_pqrs = self.integrals.ao_to_mo(self.mo_coeff,self.mo_coeff,self.mo_coeff,self.mo_coeff)
        # Initialise Hessian matrix
        hessian = np.zeros((self.nmo,self.nmo,self.nmo,self.nmo))
        # Compute Fock contributions
        for i in range(no):
            hessian[no:,i,no:,i] += 4 * Fmo[no:,no:]
        for a in range(no,self.nmo):
            hessian[a,:no,a,:no] -= 4 * Fmo[:no,:no]
        # Compute two-electron contributions
        hessian[no:,:no,no:,:no] += 16 * np.einsum('abij->aibj', eri_pqrs[no:,no:,:no,:no], optimize="optimal")
        hessian[no:,:no,no:,:no] -=  4 * np.einsum('aibj->aibj', eri_pqrs[no:,:no,no:,:no], optimize="optimal")
        hessian[no:,:no,no:,:no] -=  4 * np.einsum('aijb->aibj', eri_pqrs[no:,:no,:no,no:], optimize="optimal")
        # Return suitably shaped array
        return (hessian[:,:,self.rot_idx])[self.rot_idx,:]

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
        """Return a copy of the current RHF object"""
        them = RHF(self.integrals, verbose=self.verbose)
        them.initialise(self.mo_coeff)
        return them

    def overlap(self, them):
        """Compute the (nonorthogonal) many-body overlap with another RHF wavefunction (them)"""
        raise NotImplementedError("RHF overlap not implemented")

    def hamiltonian(self, them):
        """Compute the (nonorthogonal) many-body Hamiltonian coupling with another RHF wavefunction (them)"""
        raise NotImplementedError("RHF Hamiltonian not implemented")

    def initialise(self, mo_guess):
        """Initialise the wave function with a set of molecular orbital coefficients"""
        # Make sure orbitals are orthogonal
        self.mo_coeff = orthogonalise(mo_guess, self.integrals.overlap_matrix())
        # Update the density and Fock matrices
        self.update()

    def update(self):
        """Update the 1RDM and Fock matrix for the current state"""
        self.get_density()
        self.get_fock()

    def get_density(self):
        """Compute the 1RDM for the current state in AO basis"""
        Cocc = self.mo_coeff[:,:self.nocc]
        self.dens = np.dot(Cocc, Cocc.T)

    def get_fock(self):
        """Compute the Fock matrix for the current state"""
        self.fock = self.integrals.build_fock(self.dens)

    def diagonalise_fock(self):
        """Diagonalise the Fock matrix"""
        # Get the orthogonalisation matrix
        X = self.integrals.orthogonalization_matrix()
        # Project to linearly independent orbitals
        Ft = np.linalg.multi_dot([X.T, self.fock, X])
        # Diagonalise the Fock matrix
        self.orbital_energies, Ct = np.linalg.eigh(Ft)
        # Transform back to the original basis
        self.mo_coeff = np.dot(X, Ct)
        # Update density and Fock matrices
        self.update()

    def try_fock(self, fock):
        """Try an extrapolated Fock matrix and update the orbital coefficients"""
        self.fock = fock
        self.diagonalise_fock()

    def get_diis_error(self):
        """Compute the DIIS error vector and DIIS error"""
        err_vec  = np.linalg.multi_dot([self.fock, self.dens, self.integrals.overlap_matrix()])
        err_vec -= err_vec.T
        return err_vec.ravel(), np.linalg.norm(err_vec)

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
        """Create a matrix of boolean of size (nbsf,nbsf). 
           A True element means that this rotation should be taken into account during the optimization.
        """
        # Include only occupied-virtual rotations
        mask = np.zeros((self.nmo,self.nmo), dtype=bool)
        mask[self.nocc:,:self.nocc] = True
        return mask
    
    def get_orbital_guess(self, method="Core"):
        """Get a guess for the molecular orbital coefficients"""
        if(method == "Core"):
            # Build Fock matrix with zero density
            self.dens = np.zeros((self.nbsf,self.nbsf))
            self.get_fock()
            # Get orbital coefficients by diagonalising Fock matrix
            self.diagonalise_fock()
        else:
            raise NotImplementedError(f"Orbital guess method {method} not implemented")