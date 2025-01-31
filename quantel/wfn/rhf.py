#!/usr/bin/python3
# Author: Hugh G. A. Burton

import numpy as np
import scipy.linalg
import h5py
from quantel.utils.linalg import orthogonalise, matrix_print
from .wavefunction import Wavefunction
import quantel

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
        self.nbeta     = self.nalfa #integrals.molecule().nbeta()

        # Get number of basis functions and linearly independent orbitals
        self.nbsf      = integrals.nbsf()
        self.nmo       = integrals.nmo()
        self.with_xc    = (type(integrals) is not quantel.lib._quantel.LibintInterface)
        if(self.with_xc): self.with_xc = (integrals.xc is not None)

        # For now, we assume that the number of alpha and beta electrons are the same
        assert(self.nalfa == self.nbeta)
        self.nocc      = self.nalfa
        self.verbose   = verbose

        # Setup the indices for relevant orbital rotations
        self.rot_idx   = self.uniq_var_indices() # Indices for orbital rotations
        self.nrot      = np.sum(self.rot_idx) # Number of orbital rotations

        # Define the orbital energies and coefficients
        self.mo_coeff         = None
        self.mo_energy = None
    
    def initialise(self, mo_guess, ci_guess=None):
        """Initialise the wave function with a set of molecular orbital coefficients"""
        # Make sure orbitals are orthogonal
        self.mo_coeff = orthogonalise(mo_guess, self.integrals.overlap_matrix())
        # Update the density and Fock matrices
        self.update()

    @property
    def dim(self):
        """Get the number of degrees of freedom"""
        return self.nrot

    @property
    def energy(self):
        """Get the energy of the current RHF state"""
        # Nuclear potential
        En  = self.integrals.scalar_potential()
        # One-electron energy
        E1 = 2 * np.einsum('pq,pq', self.integrals.oei_matrix(True), self.dens, optimize="optimal")
        # Two-electron energy
        E2 = np.einsum('pq,pq', self.JK, self.dens, optimize="optimal")
        # Exchange correlation
        Exc = self.exc
        # Save components
        self.energy_components = dict(Nuclear=En, One_Electron=E1, Two_Electron=E2, Exchange_Correlation=Exc)
        return En + E1 + E2 + Exc

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

        # Get occupied and virtual orbital coefficients
        Cocc = self.mo_coeff[:,:no].copy()
        Cvir = self.mo_coeff[:,no:].copy()

        # Compute ao_to_mo integral transform
        eri_abij = self.integrals.tei_ao_to_mo(Cvir,Cvir,Cocc,Cocc,True,False)
        eri_aibj = self.integrals.tei_ao_to_mo(Cvir,Cocc,Cvir,Cocc,True,False)

        # Initialise Hessian matrix
        hessian = np.zeros((self.nmo,self.nmo,self.nmo,self.nmo))

        # Compute Fock contributions
        for i in range(no):
            hessian[no:,i,no:,i] += 4 * Fmo[no:,no:]
        for a in range(no,self.nmo):
            hessian[a,:no,a,:no] -= 4 * Fmo[:no,:no]

        # Compute two-electron contributions
        hessian[no:,:no,no:,:no] += 16 * np.einsum('abij->aibj', eri_abij, optimize="optimal")
        hessian[no:,:no,no:,:no] -=  4 * np.einsum('aibj->aibj', eri_aibj, optimize="optimal")
        hessian[no:,:no,no:,:no] -=  4 * np.einsum('abji->aibj', eri_abij, optimize="optimal")

        # Return suitably shaped array
        return (hessian[:,:,self.rot_idx])[self.rot_idx,:]

    def print(self,verbose=1):
        """ Print details about the state energy and orbital coefficients

            Inputs:
                verbose : level of verbosity
                          0 = No output
                          1 = Print energy components and spin
                          2 = Print energy components, spin, and occupied orbital coefficients
                          3 = Print energy components, spin, and all orbital coefficients
                          4 = Print energy components, spin, Fock matrix, and all orbital coefficients 
        """
        if(verbose > 0):
            print("\n ---------------------------------------------")
            print(f"         Total Energy = {self.energy:14.8f} Eh")
            for key, value in self.energy_components.items():
                print(f" {key.replace('_',' '):>20s} = {value:14.8f} Eh")
            print(" ---------------------------------------------")
            print(f"        <Sz> = {0:5.2f}")
            print(f"        <S2> = {self.s2:5.2f}")
        if(verbose > 1):
            matrix_print(self.mo_coeff[:,:self.nocc], title="Occupied Orbital Coefficients")
        if(verbose > 2):
            matrix_print(self.mo_coeff[:,self.nocc:], title="Virtual Orbital Coefficients", offset=self.nocc)
        if(verbose > 3):
            matrix_print(self.fock, title="Fock Matrix (AO basis)")
        print()

    def save_to_disk(self,tag):
        """Save object to disk with prefix 'tag'"""
        # Canonicalise orbitals
        self.canonicalize()
 
         # Save hdf5 file with MO coefficients, orbital energies, energy, and spin
        with h5py.File(tag+".hdf5", "w") as F:
            F.create_dataset("mo_coeff", data=self.mo_coeff)
            F.create_dataset("mo_energy", data=self.mo_energy)
            F.create_dataset("energy", data=self.energy)
            F.create_dataset("s2", data=self.s2)    
        
        # Save numpy txt file with energy and Hessian indices
        hindices = self.get_hessian_index()
        with open(tag+".solution", "w") as F:
            F.write(f"{self.energy:18.12f} {hindices[0]:5d} {hindices[1]:5d} {self.s2:12.6f}\n")

    def read_from_disk(self,tag):
        """Read object from disk with prefix 'tag'"""
        # Read MO coefficients from hdf5 file
        with h5py.File(tag+".hdf5", "r") as F:
            mo_read = F["mo_coeff"][:]
        # Initialise object
        self.initialise(mo_read)

    def copy(self):
        """Return a copy of the current RHF object"""
        them = RHF(self.integrals, verbose=self.verbose)
        them.initialise(self.mo_coeff)
        return them

    def overlap(self, them):
        """Compute the (nonorthogonal) many-body overlap with another RHF wavefunction (them)"""
        if(self.nocc != them.nocc):
            return 0
        nocc = self.nocc
        ovlp = self.integrals.overlap_matrix()
        S = np.linalg.multi_dot([self.mo_coeff[:,:nocc].T, ovlp, them.mo_coeff[:,:nocc]])
        return np.linalg.det(S)**2

    def hamiltonian(self, them):
        """Compute the (nonorthogonal) many-body Hamiltonian coupling with another RHF wavefunction (them)"""
        raise NotImplementedError("RHF Hamiltonian not implemented")

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
        # Compute the Coulomb and Exchange matrices
        self.fock = self.integrals.build_fock(self.dens)
        self.JK   = self.fock - self.integrals.oei_matrix(True)
        # Compute the exchange-correlation energy
        self.exc, self.vxc, NULL = self.integrals.build_vxc(self.dens, self.dens) if(self.with_xc) else 0,0,0
        self.fock += self.vxc

    def canonicalize(self):
        """Diagonalise the occupied and virtual blocks of the Fock matrix"""
        # Initialise orbital energies
        self.mo_energy = np.zeros(self.nmo)
        # Get Fock matrix in MO basis
        Fmo = np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])
        # Extract occupied and virtual blocks
        Focc = Fmo[:self.nocc,:self.nocc]
        Fvir = Fmo[self.nocc:,self.nocc:]
        # Diagonalise the occupied and virtual blocks
        self.mo_energy[:self.nocc], Qocc = np.linalg.eigh(Focc)
        self.mo_energy[self.nocc:], Qvir = np.linalg.eigh(Fvir)
        # Build the canonical MO coefficients
        self.mo_coeff[:,:self.nocc] = np.dot(self.mo_coeff[:,:self.nocc], Qocc)
        self.mo_coeff[:,self.nocc:] = np.dot(self.mo_coeff[:,self.nocc:], Qvir)
        self.update()
        # Get orbital occupation
        self.mo_occ = np.zeros(self.nmo)
        self.mo_occ[:self.nocc] = 2.0
        # Combine full transformation matrix
        Q = np.zeros((self.nmo,self.nmo))
        Q[:self.nocc,:self.nocc] = Qocc
        Q[self.nocc:,self.nocc:] = Qvir
        return Q

    def get_preconditioner(self):
        """Compute approximate diagonal of Hessian"""
        # Get Fock matrix in MO basis
        fock_mo = np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])
        # Initialise approximate preconditioner
        Q = np.zeros((self.nmo,self.nmo))
        # Include dominate generalised Fock matrix terms
        for p in range(self.nmo):
            for q in range(p):
                Q[p,q] = 2 * (fock_mo[p,p] - fock_mo[q,q])
        return np.abs(Q[self.rot_idx])

    def diagonalise_fock(self):
        """Diagonalise the Fock matrix"""
        # Get the orthogonalisation matrix
        X = self.integrals.orthogonalization_matrix()
        # Project to linearly independent orbitals
        Ft = np.linalg.multi_dot([X.T, self.fock, X])
        # Diagonalise the Fock matrix
        self.mo_energy, Ct = np.linalg.eigh(Ft)
        # Transform back to the original basis
        self.mo_coeff = np.dot(X, Ct)
        # Update density and Fock matrices
        self.update()

    def transform_vector(self,vec,step,X=None):
        """ Perform orbital rotation for vector in tangent space"""
        # Build vector in antisymmetric form
        kappa = np.zeros((self.nmo, self.nmo))
        kappa[self.rot_idx] = vec
        kappa = kappa - kappa.T
        # Only horizontal transformations leave unchanged
        if not X is None:
            kappa = kappa @ X
            kappa = X.T @ kappa
        return kappa[self.rot_idx]

    def get_variance(self):
        """ Compute the variance of the energy with respect to the current wave function
            This approach makes use of MRCISD sigma vector"""
        # Build full MO integral object
        mo_ints = quantel.MOintegrals(self.integrals)
        mo_ints.update_orbitals(self.mo_coeff,0,self.nmo)
        # Build MRCISD space
        nvir = self.nmo - self.nocc
        fulldets = [self.nocc*'2'+nvir*'0']
        mrcisd = quantel.CIspace(mo_ints,self.nmo,self.nalfa,self.nbeta)
        mrcisd.initialize('custom', fulldets)
        # Build CI vector in MRCISD space
        civec = [1]
        # Compute variance
        E, var = mrcisd.get_variance(civec)
        if(abs(E - self.energy) > 1e-12):
            raise RuntimeError("GenealogicalCSF:get_variance: Energy mismatch in variance calculation")
        
        return var

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
    
    def get_orbital_guess(self, method="gwh"):
        """Get a guess for the molecular orbital coefficients"""
        # Get one-electron integrals and overlap matrix 
        h1e = self.integrals.oei_matrix(True)
        s = self.integrals.overlap_matrix()
        
        # Build guess Fock matrix
        if(method.lower() == "core"):
            # Use core Hamiltonian as guess
            self.fock = h1e.copy()
        elif(method.lower() == "gwh"):
            # Build GWH guess Hamiltonian
            K = 1.75
            
            self.fock = np.zeros((self.nbsf,self.nbsf))
            for i in range(self.nbsf):
                for j in range(self.nbsf):
                    self.fock[i,j] = 0.5 * (h1e[i,i] + h1e[j,j]) * s[i,j]
                    if(i!=j):
                        self.fock[i,j] *= 1.75
            
        else:
            raise NotImplementedError(f"Orbital guess method {method} not implemented")
        
        # Get orbital coefficients by diagonalising Fock matrix
        self.diagonalise_fock()

    def deallocate(self):
        pass
        
    def approx_hess_on_vec(self, vec, eps=1e-3):
        """ Compute the approximate Hess * vec product using forward finite difference """
        # Get current gradient
        g0 = self.gradient.copy()
        # Save current position
        mo_save = self.mo_coeff.copy()
        # Get forward gradient
        self.take_step(eps * vec)
        g1 = self.gradient.copy()
        # Restore to origin
        self.mo_coeff = mo_save.copy()
        self.update()
        # Parallel transport back to current position
        g1 = self.transform_vector(g1, - eps * vec)
        # Get approximation to H @ sk
        return (g1 - g0) / eps