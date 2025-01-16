#!/usr/bin/python3
# Modified from ss_casscf code of Antoine Marie and Hugh G. A. Burton
# This is code for a CSF, which can be formed in a variety of ways.

import numpy as np
import scipy, quantel, h5py
from quantel.utils.csf_utils import get_shells, get_shell_exchange
from quantel.utils.linalg import orthogonalise, stable_eigh
from quantel.gnme.csf_noci import csf_coupling
from .wavefunction import Wavefunction
import time

def flag_transport(A,T,mask,max_order=50,tol=1e-4):
   tA = A.copy()
   M  = A.copy()
   for i in range(max_order):
       TM = T @ M
       M = - 0.5 * (TM - TM.T) / (i+1)
       M[mask] = 0
       if(np.max(np.abs(M)) < tol):
           break
       tA += M
   return tA

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

        # Orthogonalise the MO coefficients
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
        self.invariant  = self.invariant_indices()
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
        E -= 0.25 * np.einsum('pq,qp',self.dj,self.K[0])
        for w in range(self.nshell):
            E += 0.5 * np.einsum('pq,qp',self.K[1+w], 
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
        g1 = self.transform_vector(g1, - eps * vec)
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
        self.dj, self.dk, self.vd = self.get_density_matrices()
        # Update JK matrices (AO basis) 
        self.J, self.K = self.get_JK_matrices(self.vd)
        # Get Fock matrix (AO basis)
        self.fock = self.integrals.oei_matrix(True) + self.J - 0.5 * np.einsum('mpq->pq',self.K)
        # Get generalized Fock matrices
        self.gen_fock, self.Ipqpq, self.Ipqqp = self.get_generalised_fock()
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

    def read_from_orca(self,json_file):
        """ Read a set of CSF coefficients from ORCA gbw file.
            This requires the orca_2json executable to be available and spin_coupling 
            must be set in the Quantel input file.
        """
        import json
        # Read ORCA Json file
        with open(json_file, 'r') as f:
            data = json.load(f)
        mo_read = np.array([value['MOCoefficients'] for value in data['Molecule']['MolecularOrbitals']['MOs']]).T
        
        # TODO: For now, we have a temporary fix to change the sign of the f+3 and f-3 orbitals, 
        #       which appear to be inconsistent between Libint and ORCA
        orb_labels = data['Molecule']['MolecularOrbitals']['OrbitalLabels']
        phase_shift = []
        for i, l in enumerate(orb_labels):
            if (r'f+3' in l) or (r'f-3' in l):
                phase_shift.append(i)
        mo_read[phase_shift,:] *= -1
         
        # Initialise the wave function
        self.initialise(mo_read, spin_coupling=self.spin_coupling)   

        # Check the input
        if mo_read.shape[0] != self.nbsf:
            raise ValueError("Inccorect number of AO basis functions in file")
        if mo_read.shape[1] < self.nocc:
            raise ValueError("Insufficient orbitals in file to represent occupied orbitals")
        if mo_read.shape[1] > self.nmo:
            raise ValueError("Too many orbitals in file")

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
        n2 = self.nbsf * self.nbsf
        hcore = self.integrals.oei_matrix(True)
        eri   = self.integrals.tei_array().reshape((n2,n2))
        ovlp  = self.integrals.overlap_matrix()
        enuc  = self.integrals.scalar_potential()
        return csf_coupling(self, them, ovlp, hcore, eri, enuc)
    
    def get_orbital_guess(self, method="gwh"):
        """Get a guess for the molecular orbital coefficients"""
        h1e = self.integrals.oei_matrix(True)
        s = self.integrals.overlap_matrix()
        
        if(method.lower() == "core"):
            # Use core Hamiltonian as guess
            hguess = h1e.copy()

        elif(method.lower() == "gwh"):
            # Build GWH guess Hamiltonian
            K = 1.75
            
            hguess = np.zeros((self.nbsf,self.nbsf))
            for i in range(self.nbsf):
                for j in range(self.nbsf):
                    hguess[i,j] = 0.5 * (h1e[i,i] + h1e[j,j]) * s[i,j]
                    if(i!=j):
                        hguess[i,j] *= 1.75
            
        else:
            raise NotImplementedError(f"Orbital guess method {method} not implemented")
        
        # Solve initial generalised eigenvalue problem
        e, Cguess = scipy.linalg.eigh(hguess, s)
        self.initialise(Cguess, spin_coupling=self.spin_coupling)

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

    def transform_vector(self,vec,step,X=None):
        """ Perform orbital rotation for vector in tangent space"""
        # Construct transformation matrix
        orb_step = np.zeros((self.nmo, self.nmo))
        orb_step[self.rot_idx] = step
        orb_step = orb_step - orb_step.T

        # Build vector in antisymmetric form
        kappa = np.zeros((self.nmo, self.nmo))
        kappa[self.rot_idx] = vec
        kappa = kappa - kappa.T

        # Apply transformation
        kappa = flag_transport(kappa,orb_step,self.invariant)

        # 05/10/2024 - This is the old formula, only applicable for a small step
        #Q = scipy.linalg.expm(0.5 * (orb_step - orb_step.T))
        #kappa = kappa @ Q
        #kappa = Q.T @ kappa

        # Also apply horizontal transform
        if not X is None:
            kappa = kappa @ X
            kappa = X.T @ kappa

        # Return transformed vector
        return kappa[self.rot_idx]
    
    def get_density_matrices(self):
        """ Compute total density matrix and relevant matrices for K build"""
        # Number of densities (core + open-shell)
        nopen = self.nocc-self.ncore
        # Initialise densities
        vd = np.zeros((1+nopen,self.nbsf,self.nbsf))
        # Core contribution       
        vd[0] = 2 * self.mo_coeff[:,:self.ncore] @ self.mo_coeff[:,:self.ncore].T
        # Contribution from each active orbital
        for id, i in enumerate(range(self.ncore,self.nocc)):
            vd[id+1] = self.mo_occ[i] * np.outer(self.mo_coeff[:,i],self.mo_coeff[:,i])

        # Extract shell densities
        dj  = np.einsum('kpq->pq',vd)
        dk = np.zeros((self.nshell+1,self.nbsf,self.nbsf))
        dk[0] = vd[0].copy()
        for Ishell in range(self.nshell):
            shell = [1+i-self.ncore for i in self.shell_indices[Ishell]]
            dk[Ishell+1] += np.einsum('vpq->pq',vd[shell])
        return dj, dk, vd

    def get_JK_matrices(self, vd):
        ''' Compute the JK matrices and diagonal two-electron integrals

            This function also computes the density matrices (maybe redundant)

            Input:
                vd: Density matrices for core and each open orbital

            Returns:
                J: Total Coulomb matrix
                K: Exchange matrices for each shell
                Ipqpq: Diagonal elements of J matrix
                Ipqqp: Diagonal elements of K matrix
        '''
        # Number of densities (core + open-shell)
        nopen = vd.shape[0]-1

        # Call integrals object to build J and K matrices
        self.vJ, self.vK = self.integrals.build_multiple_JK(vd,vd,nopen+1,nopen+1)

        # Get the total J matrix
        J = np.einsum('kpq->pq',self.vJ)
        # Get exchange matrices for each shell
        K = np.zeros((self.nshell+1,self.nbsf,self.nbsf))
        K[0] = self.vK[0].copy()
        for Ishell in range(self.nshell):
            shell = [1+i-self.ncore for i in self.shell_indices[Ishell]]
            K[Ishell+1] += np.einsum('vpq->pq',self.vK[shell])

        return J, K

    def get_generalised_fock(self):
        """ Compute the generalised Fock matrix in AO basis"""
        # Initialise memory
        F = np.zeros((self.nmo, self.nmo))

        # Memory for diagonal elements
        self.gen_fock_diag = np.zeros((self.nmo,self.nmo))
        # Core contribution
        Fcore_ao = 2*(self.integrals.oei_matrix(True) + self.J 
                      - 0.5 * np.sum(self.K[i] for i in range(self.nshell+1)))
        # AO-to-MO transformation
        Fcore_mo = np.linalg.multi_dot([self.mo_coeff.T, Fcore_ao, self.mo_coeff])
        for i in range(self.ncore):
            self.gen_fock_diag[i,:] = Fcore_mo.diagonal()
        F[:self.ncore,:] = Fcore_mo[:self.ncore,:]

        # Open-shell contributions
        for W in range(self.nshell):
            # Get shell indices and coefficients
            shell = self.shell_indices[W]
            # One-electron matrix, Coulomb and core exchange
            Fw_ao = self.integrals.oei_matrix(True) + self.J - 0.5 * self.K[0]
            # Different shell exchange
            Fw_ao += np.einsum('v,vpq->pq',self.beta[W],self.K[1:])
            # AO-to-MO transformation
            Fw_mo = np.linalg.multi_dot([self.mo_coeff.T, Fw_ao, self.mo_coeff])
            for w in shell:
                self.gen_fock_diag[w,:] = Fw_mo.diagonal()
            F[shell,:] = Fw_mo[shell,:]
        
        # Get diagonal J/K terms
        nopen = self.nocc-self.ncore
        Ipqpq = np.zeros((nopen,self.nmo))
        Ipqqp = np.zeros((nopen,self.nmo))
        for i in range(nopen):
            Ji = self.mo_coeff.T @ self.vJ[1+i] @ self.mo_coeff
            Ki = self.mo_coeff.T @ self.vK[1+i] @ self.mo_coeff
            Ipqpq[i] = np.diag(Ji)
            Ipqqp[i] = np.diag(Ki)

        return F, Ipqpq, Ipqqp

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
        vKmn = np.einsum('pm,wpq,qn->wmn',self.mo_coeff, self.K, self.mo_coeff)
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
        # Initialise approximate preconditioner
        Q = np.zeros((self.nmo,self.nmo))

        # Include dominate generalised Fock matrix terms
        for p in range(self.nmo):
            for q in range(p):
                Q[p,q] = 2 * ( (self.gen_fock_diag[p,q] - self.gen_fock_diag[q,q]) 
                             + (self.gen_fock_diag[q,p] - self.gen_fock_diag[p,p]) )
           
        # Compute two-electron corrections involving active orbitals
        Acoeff = self.Ipqqp
        for q in range(self.ncore,self.nocc):
            for p in range(q):
                Q[q,p] += 4 * (self.mo_occ[p]-self.mo_occ[q])**2 * Acoeff[q-self.ncore,p]
            for p in range(q+1,self.nmo):
                Q[p,q] += 4 * (self.mo_occ[p]-self.mo_occ[q])**2 * Acoeff[q-self.ncore,p]

        Bcoeff = self.Ipqpq + self.Ipqqp
        for W in range(self.nshell):
            for q in self.shell_indices[W]:
                # Core-Active
                for p in range(self.ncore):
                    Q[q,p] -= 2 * Bcoeff[q-self.ncore,p]
                # Active-Active
                for V in range(W):
                    for p in self.shell_indices[V]:
                        Q[q,p] -= 4 * (1 + self.beta[V,W]) * Bcoeff[q-self.ncore,p]
                # Virtual-Active
                for p in range(self.nocc,self.nmo):
                    Q[p,q] -= 2 * (self.mo_occ[p] + self.mo_occ[q]) * Bcoeff[q-self.ncore,p]

        # Return absolute preconditioner
        return np.abs(Q[self.rot_idx])

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

    def invariant_indices(self):
        """ This function creates a matrix of boolean of size (norb,norb).
            A True element means that this rotation should be taken into
            account during the optimization. Taken from pySCF.mcscf.casscf
        """
        mask = np.zeros((self.nmo, self.nmo), dtype=bool)
        mask[:self.ncore,:self.ncore] = True
        for W in self.shell_indices:
            mask[W,W] = True
        mask[self.nocc:,self.nocc:] = True
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
        Forms the canonicalised MO coefficients by diagonalising invariant 
        subblocks of the Fock matrix
        """
        # Transform Fock matrix to MO basis
        fock = np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])

        # Initialise transformation matrix
        self.mo_energy = np.zeros(self.nmo)
        Q = np.zeros((self.nmo,self.nmo))

        # Get core transformation
        foo = fock[self.core_indices,:][:,self.core_indices]
        self.mo_energy[:self.ncore], Qoo = stable_eigh(foo)
        for i, ii in enumerate(self.core_indices):
            for j, jj in enumerate(self.core_indices):
                Q[ii,jj] = Qoo[i,j]

        # Loop over shells
        for W in self.shell_indices:
            fww = fock[W,:][:,W]
            self.mo_energy[W], Qww = stable_eigh(fww)
            for i, ii in enumerate(W):
                for j, jj in enumerate(W):
                    Q[ii,jj] = Qww[i,j]

        # Virtual transformation
        fvv = fock[self.nocc:, self.nocc:]
        self.mo_energy[self.nocc:], Qvv = stable_eigh(fvv)
        Q[self.nocc:,self.nocc:] = Qvv

        # Apply transformation
        if(np.linalg.det(Q) < 0):
            Q[:,0] *= -1
        self.mo_coeff = self.mo_coeff @ Q
        
        # Update generalised Fock matrix and diagonal approximations
        #self.update_integrals()
        self.gen_fock, self.Ipqpq, self.Ipqqp = self.get_generalised_fock()
        return Q
