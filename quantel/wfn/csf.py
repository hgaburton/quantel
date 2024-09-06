#!/usr/bin/python3
# Modified from ss_casscf code of Antoine Marie and Hugh G. A. Burton
# This is code for a CSF, which can be formed in a variety of ways.

import numpy as np
import scipy, quantel, h5py, warnings
from quantel.utils.csf_utils import get_csf_vector, get_shells, get_shell_exchange
from quantel.utils.linalg import orthogonalise
from quantel.gnme.csf_noci import csf_coupling
from .wavefunction import Wavefunction
import time


class GenealogicalCSF(Wavefunction):
    """ 
        A class for a Configuration State Function (CSF) using the genealogical coupling pattern.

        Inherits from the Wavefunction abstract base class with pure virtual properties:
            - energy
            - gradient
            - hessian
            - take_step
            - save_last_step
            - restore_step
    """
    def __init__(self, integrals, spin_coupling, verbose=0):
        """ Initialise the CSF wave function
                integrals     : quantel integral interface
                spin_coupling : genealogical coupling pattern
                verbose       : verbosity level
        """
        if(spin_coupling == 'cs'):
            self.spin_coupling = ''
        else:
            self.spin_coupling = spin_coupling

        self.verbose       = verbose
        # Initialise integrals object
        self.integrals  = integrals
        self.nalfa      = integrals.molecule().nalfa()
        self.nbeta      = integrals.molecule().nbeta()
        # Initialise molecular integrals object
        self.mo_ints    = quantel.MOintegrals(integrals)
        # Get number of basis functions and linearly independent orbitals
        self.nbsf       = integrals.nbsf()
        self.nmo        = integrals.nmo()
    
    
    def sanity_check(self):
        '''Need to be run at the start of the kernel to verify that the number of 
           orbitals and electrons in the CAS are consistent with the system '''
        # Check number of active orbitals is positive
        if self.cas_nmo < 0:
            raise ValueError("Number of active orbitals must be positive")
        # Check number of active electrons doesn't exceed total number of electrons
        if(self.cas_nalfa > self.cas_nmo or self.cas_nbeta > self.cas_nmo):
            raise ValueError("Number of active electrons must be <= number of active orbitals")
        # Check number of active electrons doesn't exceed total number of electrons
        if(self.cas_nalfa > self.nalfa or self.cas_nbeta > self.nbeta):
            raise ValueError("Number of active electrons must be <= total number of electrons")
        # Check number of occupied orbitals doesn't exceed total number of orbitals
        if(self.nocc > self.nmo):
            raise ValueError("Number of inactive and active orbitals must be <= total number of orbitals")
                             

    def initialise(self, mo_guess, spin_coupling=None, mat_ci=None, integrals=True):
        """ Initialise the CSF object with a set of MO coefficients"""
        if(spin_coupling is None):
            spin_coupling = self.spin_coupling
        if(spin_coupling == 'cs'):
            spin_coupling = ''
        # Save orbital coefficients
        mo_guess      = orthogonalise(mo_guess, self.integrals.overlap_matrix())
        if(mo_guess.shape[1] != self.nmo):
            raise ValueError("Number of orbitals in MO coefficient matrix is incorrect")
        self.mo_coeff = mo_guess

        # Get active space definition
        self.cas_nmo    = len(spin_coupling)
        self.cas_nalfa  = sum(int(s=='+') for s in spin_coupling)
        self.cas_nbeta  = sum(int(s=='-') for s in spin_coupling)
        # Get number of core electrons
        self.ncore = self.integrals.molecule().nelec() - self.cas_nalfa - self.cas_nbeta
        if(self.ncore % 2 != 0):
            raise ValueError("Number of core electrons must be even")
        if(self.ncore < 0):
            raise ValueError("Number of core electrons must be positive")
        self.ncore = self.ncore // 2
        self.nalfa = self.ncore + self.cas_nalfa
        self.nbeta = self.ncore + self.cas_nbeta
        # Get numer of 'occupied' orbitals
        self.nocc = self.ncore + self.cas_nmo
        self.sanity_check()

        # Get determinant list and CSF occupation/coupling vectors
        self.spin_coupling = spin_coupling
        self.core_indices, self.shell_indices = get_shells(self.ncore,self.spin_coupling)
        self.mo_occ = np.zeros(self.nmo)
        self.mo_occ[:self.nocc] = 2
        self.mo_occ[self.ncore:self.nocc] = 1
        # Get information about the electron shells
        self.beta = get_shell_exchange(self.ncore,self.shell_indices, self.spin_coupling)
        self.nshell = len(self.shell_indices)

        # Save mapping indices for unique orbital rotations
        self.frozen     = None
        self.rot_idx    = self.uniq_var_indices(self.frozen)
        self.nrot       = np.sum(self.rot_idx)

        # Initialise integrals
        if (integrals): self.update_integrals()

    def deallocate(self):
        pass

    @property
    def dim(self):
        """Number of degrees of freedom"""
        return self.nrot

    @property
    def energy(self):
        """ Compute the energy corresponding to a given set of
             one-el integrals, two-el integrals, 1- and 2-RDM
        """
        # Nuclear repulsion
        E = self.integrals.scalar_potential()
        # One-electron energy
        E += np.einsum('pq,qp',self.dj,self.integrals.oei_matrix(True))
        # Coulomb energy
        E += 0.5 * np.einsum('pq,qp',self.dj,self.J)
        # Exchange energy
        E -= 0.25 * np.einsum('pq,qp',self.dj,self.vK[0])
        for w in range(self.nshell):
            E += 0.5 * np.einsum('pq,qp',self.vK[1+w], 
                          np.einsum('v,vpq->pq',self.beta[w],self.dk[1:]) - 0.5 * self.dk[0])
        return E

    @property
    def sz(self):
        """<S_z> value of the current wave function"""
        return 0.5 * np.sum([1 if s=='+' else -1 for s in self.spin_coupling])

    @property
    def s2(self):
        """ <S^2> value of the current wave function
            Uses the formula S^2 = S- S+ + Sz Sz + Sz, which corresponds to 
                <S^2> = <Sz> * (<Sz> + 1) + <Nb> - sum_pq G^{ab}_{pqqp} 
            where G^{ab}_{pqqp} is the alfa-beta component of the 2-RDM
        """
        ms = np.sum([0.5 if s=='+' else -0.5 for s in self.spin_coupling])
        return self.sz * (self.sz + 1)

    @property
    def gradient(self):
        """ Compute the gradient of the energy with respect to the orbital rotations"""
        return 2 * (self.gen_fock.T - self.gen_fock)[self.rot_idx]

    @property
    def hessian(self):
        ''' This method finds orb-orb part of the Hessian '''
        # Get generalised Fock and symmetrise
        F = self.gen_fock + self.gen_fock.T

        # Get one-electron matrix elements 
        h1e = np.linalg.multi_dot([self.mo_coeff.T, self.integrals.oei_matrix(True), self.mo_coeff])

        # Combine intermediates (Eq. 10.8.53 in Helgaker book) 
        Hess = 2 * self.get_Y_intermediate()
        for i in range(self.nmo):
            Hess[i,:,i,:] += 2 * self.mo_occ[i] * h1e
            Hess[:,i,:,i] -= F
       
        # Apply permutation symmetries
        Hess = Hess - Hess.transpose(1,0,2,3)
        Hess = Hess - Hess.transpose(0,1,3,2)

        # Reshape and return
        return (Hess[:, :, self.rot_idx])[self.rot_idx, :]

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
        self.update_integrals()

        # Parallel transport back to current position
        g1 = self.transform_vector(g1, -0.5 * eps * vec)

        # Get approximation to H @ sk
        return (g1 - g0) / eps

    def hess_on_vec(self, vec):
        return self.hessian @ vec

    def get_rdm12(self):
        """ Compute the 1- and 2-electron reduced matrices from the shell coupling in occupied space
            returns: 
                dm1: 1-electron reduced density matrix
                dm2: 2-electron reduced density matrix
        """
        # Numbers 
        nocc = self.nocc
        ncore = self.ncore

        # 1-RDM
        dm1 = np.diag(self.mo_occ[:nocc])

        # 2-RDM
        dm2 = np.zeros((nocc,nocc,nocc,nocc))
        for p in range(ncore):
            for q in range(ncore):
                if(p==q):
                    dm2[p,p,p,p] = 2
                else:
                    dm2[p,q,p,q] = 4
                    dm2[p,q,q,p] = - 2
            for w in range(ncore,nocc):
                dm2[p,w,p,w] = 2
                dm2[p,w,w,p] = -1
                dm2[w,p,w,p] = 2
                dm2[w,p,p,w] = -1

        for W, sW in enumerate(self.shell_indices):
            for V, sV in enumerate(self.shell_indices):
                for w in sW:
                    for v in sV:
                        if(w==v):
                            dm2[w,w,w,w] = 0
                        else:
                            dm2[w,v,w,v] = 1 
                            dm2[w,v,v,w] = self.beta[W,V]
        return dm1, dm2
    
    def update_integrals(self):
        """ Update the integrals with current set of orbital coefficients"""
        # Update density matrices (AO basis)
        self.dj, self.dk = self.get_density_matrices()
        # Update JK matrices (AO basis) 
        self.J, self.vK = self.get_JK_matrices(self.dj,self.dk)
        # Get Fock matrix (AO basis)
        self.fock = self.integrals.oei_matrix(True) + self.J - 0.5 * np.einsum('mpq->pq',self.vK)
        # Get generalized Fock matrices
        self.gen_fock = self.get_generalised_fock()
        return 

    def save_to_disk(self, tag):
        """Save a CSF to disk with prefix 'tag'"""
        # Save hdf5 file with mo coefficients and spin coupling
        with h5py.File(tag+'.hdf5','w') as F:
            F.create_dataset("mo_coeff", data=self.mo_coeff[:,:self.nocc])
            F.create_dataset("spin_coupling", data=self.spin_coupling)
            F.create_dataset("energy", data=self.energy)
            F.create_dataset("s2", data=self.s2)
        
        # Save numpy txt file with energy and Hessian index
        hindices = self.hess_index
        with open(tag+".solution", "w") as F:
            F.write(f"{self.energy:18.12f} {hindices[0]:5d} {hindices[1]:5d} {self.s2:12.6f} {self.spin_coupling:s}\n")
        return 
    
    def read_from_disk(self, tag):
        """Read a CSF wavefunction from disk with prefix 'tag'"""
        with h5py.File(tag+'.hdf5','r') as F:
            mo_read = F['mo_coeff'][:]
            spin_coupling = str(F['spin_coupling'][...])[2:-1]

        # Initialise the wave function
        self.initialise(mo_read, spin_coupling=spin_coupling)        
        
        # Check the input
        if mo_read.shape[0] != self.nbsf:
            raise ValueError("Inccorect number of AO basis functions in file")
        if mo_read.shape[1] < self.nocc:
            raise ValueError("Insufficient orbitals in file to represent occupied orbitals")
        if mo_read.shape[1] > self.nmo:
            raise ValueError("Too many orbitals in file")
        return

    def copy(self):
        """Return a copy of the current object"""
        newcsf = GenealogicalCSF(self.integrals, self.spin_coupling, verbose=self.verbose)
        newcsf.initialise(self.mo_coeff,spin_coupling=self.spin_coupling)
        return newcsf

    def overlap(self, them):
        """ Compute the overlap between two CSF objects
        """
        ovlp = self.integrals.overlap_matrix()
        return csf_coupling(self, them, ovlp)[0]

    def hamiltonian(self, them):
        """ Compute the Hamiltonian coupling between two CSF objects
        """
        hcore = self.integrals.oei_matrix(True)
        eri   = self.integrals.tei_array(True,False).transpose(0,2,1,3).reshape(self.nbsf**2,self.nbsf**2)
        ovlp  = self.integrals.overlap_matrix()
        enuc  = self.integrals.scalar_potential()
        return csf_coupling(self, them, ovlp, hcore, eri, enuc)

    def restore_last_step(self):
        """ Restore MO coefficients to previous step"""
        self.mo_coeff = self.mo_coeff_save.copy()
        self.update_integrals()

    def save_last_step(self):
        """ Save MO coefficients"""
        self.mo_coeff_save = self.mo_coeff.copy()

    def take_step(self, step):
        """ Take a step in the orbital space"""
        self.save_last_step()
        self.rotate_orb(step[:self.nrot])
        self.update_integrals()

    def rotate_orb(self, step):
        """ Rotate molecular orbital coefficients with a step"""
        orb_step = np.zeros((self.nmo, self.nmo))
        orb_step[self.rot_idx] = step
        self.mo_coeff = np.dot(self.mo_coeff, scipy.linalg.expm(orb_step - orb_step.T))

    def transform_vector(self,vec,step):
        """ Perform orbital rotation for vector in tangent space"""
        # Construct transformation matrix
        orb_step = np.zeros((self.nmo, self.nmo))
        orb_step[self.rot_idx] = step
        Q = scipy.linalg.expm(orb_step - orb_step.T)

        # Build vector in antisymmetric form
        kappa = np.zeros((self.nmo, self.nmo))
        kappa[self.rot_idx] = vec
        kappa = kappa - kappa.T

        # Apply transformation
        kappa = kappa @ Q
        kappa = Q.T @ kappa
        # Return transformed vector
        return kappa[self.rot_idx]

    def get_density_matrices(self):
        """ Compute total density matrix and relevant matrices for K build"""
        # Total density for J matrix. Initialise with core contribution
        dj = 2 * self.mo_coeff[:,:self.ncore] @ self.mo_coeff[:,:self.ncore].T
        # Shell densities for K matrix
        dk = np.zeros((self.nshell+1,self.nbsf,self.nbsf))
        dk[0] = dj.copy()
        # Loop over shells
        for Ishell in range(self.nshell):
            shell    = self.shell_indices[Ishell]
            dk[Ishell+1] = self.mo_occ[shell[0]] * self.mo_coeff[:,shell] @ self.mo_coeff[:,shell].T
            dj += dk[Ishell+1]
        return dj, dk

    def get_JK_matrices(self,dj,dk):
        ''' Compute the JK matrices'''
        # Call integrals object to build J and K matrices
        J, vK = self.integrals.build_multiple_JK(dj,dk,1+self.nshell)        
        return J, vK

    def get_generalised_fock(self):
        """ Compute the generalised Fock matrix"""
        # Initialise memory
        F = np.zeros((self.nmo, self.nmo))

        # Memory for diagonal elements
        self.gen_fock_diag = np.zeros((self.nmo,self.nmo))
        
        # Core contribution
        Fcore_ao = 2*(self.integrals.oei_matrix(True) + self.J 
                      - 0.5 * np.sum(self.vK[i] for i in range(self.nshell+1)))
        # AO-to-MO transformation
        Ccore = self.mo_coeff[:,:self.ncore]
        Fcore_mo = np.linalg.multi_dot([self.mo_coeff.T, Fcore_ao, self.mo_coeff])
        for i in range(self.ncore):
            self.gen_fock_diag[i,:] = Fcore_mo.diagonal()
        F[:self.ncore,:] = Fcore_mo[:self.ncore,:]

        # Open-shell contributions
        for W in range(self.nshell):
            # Get shell indices and coefficients
            shell = self.shell_indices[W]
            Cw = self.mo_coeff[:,shell]
            # One-electron matrix, Coulomb and core exchange
            Fw_ao = self.integrals.oei_matrix(True) + self.J - 0.5 * self.vK[0]
            # Different shell exchange
            Fw_ao += np.einsum('v,vpq->pq',self.beta[W],self.vK[1:])
            # AO-to-MO transformation
            Fw_mo = np.linalg.multi_dot([self.mo_coeff.T, Fw_ao, self.mo_coeff])
            for w in shell:
                self.gen_fock_diag[w,:] = Fw_mo.diagonal()
            F[shell,:] = Fw_mo[shell,:]
        
        return F

    def get_Y_intermediate(self):
        """ Compute the Y intermediate required for Hessian evaluation
        """
        # Get required constants
        nmo   = self.nmo
        ncore = self.ncore
        nocc = self.nocc

        # Get required two-electron MO integrals
        Cocc = self.mo_coeff[:,:nocc].copy()
        ppoo = self.integrals.tei_ao_to_mo(self.mo_coeff,self.mo_coeff,Cocc,Cocc,True,False)
        popo = self.integrals.tei_ao_to_mo(self.mo_coeff,Cocc,self.mo_coeff,Cocc,True,False)

        # K and J in MO basis
        Jmn  = np.einsum('pm,pq,qn->mn',self.mo_coeff, self.J, self.mo_coeff)
        vKmn = np.einsum('pm,wpq,qn->wmn',self.mo_coeff, self.vK, self.mo_coeff)
        Kmn  = np.einsum('wpq->pq', vKmn)

        # Build Ypqrs
        Y = np.zeros((nmo,nmo,nmo,nmo))
        # Y_imjn
        Y[:ncore,:,:ncore,:] += 8 * np.einsum('mnij->imjn',ppoo[:,:,:ncore,:ncore]) 
        Y[:ncore,:,:ncore,:] -= 2 * np.einsum('mnji->imjn',ppoo[:,:,:ncore,:ncore])
        Y[:ncore,:,:ncore,:] -= 2 * np.einsum('mjni->imjn',popo[:,:ncore,:,:ncore])
        for i in range(ncore):
            Y[i,:,i,:] += 2 * Jmn - Kmn

        # Y_imwn
        Y[:ncore,:,ncore:nocc,:] = (4 * ppoo[:,:,:ncore,ncore:nocc].transpose(2,0,3,1)
                                      - ppoo[:,:,ncore:nocc,:ncore].transpose(3,0,2,1)
                                      - popo[:,ncore:nocc,:,:ncore].transpose(3,0,1,2))
        Y[ncore:nocc,:,:ncore,:] = Y[:ncore,:,ncore:nocc,:].transpose(2,3,0,1)

        # Y_wmvn
        for W in range(self.nshell):
            wKmn = np.einsum('v,vmn->mn',self.beta[W], vKmn[1:])
            for V in range(W,self.nshell):
                for w in self.shell_indices[W]:
                    for v in self.shell_indices[V]:
                        Y[w,:,v,:] = 2 * ppoo[:,:,w,v] + self.beta[W,V] * (ppoo[:,:,v,w] + popo[:,v,:,w])
                        if(w==v):
                            Y[w,:,w,:] = Y[w,:,w,:] + Jmn - 0.5 * vKmn[0] + wKmn
                        else:
                            Y[v,:,w,:] = Y[w,:,v,:].T
        return Y

    def get_preconditioner(self):
        """Compute approximate diagonal of Hessian"""
        fock = np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])
        Q = np.zeros((self.nmo,self.nmo))
        
        for p in range(self.nmo):
            for q in range(self.nmo):
                Q[p,q] = 2 * ( (self.gen_fock_diag[p,q] - self.gen_fock_diag[q,q]) 
                             + (self.gen_fock_diag[q,p] - self.gen_fock_diag[p,p]) )

        return Q[self.rot_idx]

    def edit_mask_by_gcoupling(self, mask):
        r"""
        This function looks at the genealogical coupling scheme and modifies a given mask.
        The mask restricts the number of free parameters.

        The algorithm works by looking at each column and traversing downwards the columns.
        """
        g_coupling_arr = list(self.spin_coupling)
        n_dim = len(g_coupling_arr)
        for i, gfunc in enumerate(g_coupling_arr):  # This is for the columns
            for j in range(i + 1, n_dim):  # This is for the rows
                if gfunc == g_coupling_arr[j]:
                    mask[j, i] = False
                else:
                    break
        return mask


    def uniq_var_indices(self, frozen):
        """ This function creates a matrix of boolean of size (norb,norb).
            A True element means that this rotation should be taken into
            account during the optimization. Taken from pySCF.mcscf.casscf
        """
        mask = np.zeros((self.nmo, self.nmo), dtype=bool)
        # Active-core rotations
        mask[self.ncore:self.nocc, :self.ncore] = True
        # Virtual-Core and Virtual-Active rotations
        mask[self.nocc:, :self.nocc] = True
        # Active-Active rotations
        mask[self.ncore:self.nocc, self.ncore:self.nocc] = np.tril(
            np.ones((self.cas_nmo, self.cas_nmo), dtype=bool), k=-1)
        
        # Modify for genealogical coupling
        if self.spin_coupling is not None:
            mask[self.ncore:self.nocc, self.ncore:self.nocc] = self.edit_mask_by_gcoupling(
                mask[self.ncore:self.nocc,self.ncore:self.nocc])
            
        # Account for any frozen orbitals   
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[:frozen] = mask[:, :frozen] = False
            else:
                frozen = np.asarray(frozen)
                mask[frozen] = mask[:, frozen] = False
        return mask


    def canonicalize(self):
        """
        Forms the canonicalised MO coefficients by diagonalising invariant subblocks of the Fock matrix
        """
        # Transform Fock matrix to MO basis
        fock = np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])

        # Get occ-occ and vir-vir blocks of (pseudo) Fock matrix
        foo = fock[:self.ncore, :self.ncore]
        faa = fock[self.ncore:self.nocc, self.ncore:self.nocc]
        fvv = fock[self.nocc:, self.nocc:]

        # Get transformations
        self.mo_energy = np.zeros(self.nmo)
        self.mo_energy[:self.ncore], Qoo = np.linalg.eigh(foo)
        self.mo_energy[self.nocc:], Qvv = np.linalg.eigh(fvv)
        self.mo_energy[self.ncore:self.nocc] = np.diag(faa)

        # Apply transformations
        self.mo_coeff[:,:self.ncore] = np.dot(self.mo_coeff[:,:self.ncore], Qoo)
        self.mo_coeff[:,self.nocc:] = np.dot(self.mo_coeff[:,self.nocc:], Qvv)
        
        # Update integrals
        self.update_integrals()

        return
