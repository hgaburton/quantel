#!/usr/bin/python3
# Author: Hugh G. A. Burton

import numpy as np
import scipy.linalg
import quantel
from quantel.utils.linalg import orthogonalise
#from exelsis.gnme.esmf_noci import esmf_coupling
from .wavefunction import Wavefunction

class ESMF(Wavefunction):
    """ Excited-state mean-field method

        Inherits from the Wavefunction abstract base class with pure virtual properties:
            - energy
            - gradient
            - hessian 
            - take_step
            - save_last_step
            - restore_step
    """

    def __init__(self, integrals, spin=0, with_ref=False, verbose=0):
        """ Initialise excited-state mean-field wavefunction
                integrals : quantel integral interface
               
        """
        # Initialise integrals object
        self.integrals = integrals
        self.nalfa     = integrals.molecule().nalfa()
        self.nbeta     = integrals.molecule().nbeta()
        # Initialise molecular integrals object
        self.mo_ints   = quantel.MOintegrals(integrals)
        # Get number of basis functions and linearly independent orbitals
        self.nbsf      = integrals.nbsf()
        self.nmo       = integrals.nmo()
        # Initialise CI space
        cistr          = ('ESMF' if with_ref else 'CIS')
        self.cispace   = quantel.CIspace(self.mo_ints, self.nmo, self.nalfa, self.nbeta, cistr)
        self.cispace.print()

        # For now, only support alpha = nbeta
        if(self.nalfa != self.nbeta):
            raise ValueError("Alpha and beta electrons must be equal")
        # Get number of determinants and spin
        self.with_ref  = with_ref
        self.spin      = spin
        self.ndet      = self.nalfa * (self.nmo - self.nalfa)
        if self.with_ref: self.nDet += 1

        self.verbose   = verbose

        # Save mapping indices for unique orbital rotations
        self.frozen     = None
        self.rot_idx    = self.uniq_var_indices(self.frozen)
        self.nrot       = np.sum(self.rot_idx)

    def initialise(self, mo_guess, ci_guess, integrals=True):
        # Save orbital coefficients
        mo_guess = orthogonalise(mo_guess, self.integrals.overlap_matrix())
        self.mo_coeff = mo_guess
        assert(self.nmo == self.mo_coeff.shape[1])

        if(self.spin==1 and self.with_ref and abs(ci_guess[0])>1e-12):
            raise ValueError("Reference determinant must have zero CI coefficient for triplet spin")

        # Orthogonalise and save CI coefficients
        ci_guess = orthogonalise(ci_guess, np.identity(self.ndet)) 
        self.mat_ci = ci_guess

        # Initialise integrals
        if(integrals): self.update_integrals()

    @property
    def dim(self):
        """Get the number of degrees of freedom"""
        return self.nrot + self.ndet - 1

    @property
    def energy(self):
        ''' Compute the energy corresponding to a given set of
             one-el integrals, two-el integrals, 1- and 2-RDM '''
        en = self.integrals.scalar_potential()
        en += np.einsum('pq,pq', self.h1e, self.dm1, optimize="optimal")
        en += 0.5 * np.einsum('pqrs,pqrs', self.h2e, self.dm2, optimize="optimal")
        return en

    @property
    def s2(self):
        ''' Compute the spin of a given FCI vector '''
        # TODO Need to implement s2 computation
        return 0 #self.fcisolver.spin_square(self.mat_ci[:,0], self.ncas, self.nelecas)[0]

    @property
    def gradient(self):
        g_orb = self.get_orbital_gradient()
        g_ci  = self.get_ci_gradient()  
        # Unpack matrices/vectors accordingly
        return np.concatenate((g_orb, g_ci))
    
    def get_orbital_gradient(self):
        """Compute the orbital component of the energy gradient"""
        g_orb = 2 * self.get_gen_fock(self.mat_ci[:,0], self.mat_ci[:,0], False)
        return (g_orb.T - g_orb)[self.rot_idx]


    def get_ci_gradient(self):
        """Compute the CI component of the energy gradient"""
        if(self.ndet > 1):
            return 2.0 * self.mat_ci[:,1:].T.dot(self.sigma)
        else:
            return np.zeros((0))

    @property
    def hessian(self):
        ''' This method concatenate the orb-orb, orb-CI and CI-CI part of the Hessian '''
        H_OrbOrb = (self.get_hessianOrbOrb()[:,:,self.rot_idx])[self.rot_idx,:]
        np.set_printoptions(linewidth=10000)
        H_CICI   = self.get_hessianCICI()
        H_OrbCI  = self.get_hessianOrbCI()[self.rot_idx,:]

        return np.block([[H_OrbOrb, H_OrbCI],
                         [H_OrbCI.T, H_CICI]])

    def restore_last_step(self):
        # Restore coefficients
        self.mo_coeff = self.mo_coeff_save.copy()
        self.mat_ci   = self.mat_ci_save.copy()
        # Finally, update our integrals for the new coefficients
        self.update_integrals()

    def save_last_step(self):
        self.mo_coeff_save = self.mo_coeff.copy()
        self.mat_ci_save   = self.mat_ci.copy()


    def take_step(self,step):
        # Save our last position
        self.save_last_step()

        # Take steps in orbital and CI space
        self.rotate_orb(step[:self.nrot])
        self.rotate_ci(step[self.nrot:])

        # Finally, update our integrals for the new coefficients
        self.canonicalize()
        self.update_integrals()


    def rotate_orb(self,step): 
        '''Rotate the molecular orbital coefficients'''
        # Transform the step into correct structure
        orb_step = np.zeros((self.nmo,self.nmo))
        orb_step[self.rot_idx] = step
        self.mo_coeff = np.dot(self.mo_coeff, scipy.linalg.expm(orb_step - orb_step.T))


    def rotate_ci(self,step): 
        """Take rotation step in the CIS space"""
        S       = np.zeros((self.ndet,self.ndet))
        S[1:,0] = step
        self.mat_ci = np.dot(self.mat_ci, scipy.linalg.expm(S - S.T))

    def save_to_disk(self,tag):
        """Save object to disk with prefix 'tag'"""
        # Get the Hessian index
        hindices = self.get_hessian_index()

        # Save coefficients, CI, and energy
        np.savetxt(tag+'.mo_coeff', self.mo_coeff, fmt="% 20.16f")
        np.savetxt(tag+'.mat_ci',   self.mat_ci, fmt="% 20.16f")
        np.savetxt(tag+'.energy',   
                   np.array([[self.energy, hindices[0], hindices[1], self.s2]]), 
                   fmt="% 18.12f % 5d % 5d % 12.6f")


    def read_from_disk(self,tag):
        """Read object from disk with prefix 'tag'"""
        # Read MO coefficient and CI coefficients
        mo_coeff = np.genfromtxt(tag+".mo_coeff")
        ci_coeff = np.genfromtxt(tag+".mat_ci")

        # Initialise object
        self.initialise(mo_coeff, ci_coeff)

    def copy(self):
        # Return a copy of the current object
        newcas = ESMF(self.mol, spin=self.spin, ref_allowed=self.with_ref)
        newcas.initialise(self.mo_coeff, self.mat_ci, integrals=False)
        return newcas
    

    def update_integrals(self):
        # Update integral object
        self.mo_ints.update_orbitals(self.mo_coeff,0,self.nmo)

        # One-electron Hamiltonian
        self.h1e = self.mo_ints.oei_matrix(True)
        # Two-electron MO integrals orbitals (physicists notation)
        self.h2e = self.mo_ints.tei_array(True,False).transpose(0,2,1,3)
        # Reduced density matrices 
        self.dm1, self.dm2 = self.get_rdm12()

        # Fock matrix for reference determinant
        self.ref_fock = self.h1e + (2 * np.einsum('pqjj->pq',self.h2e[:,:,:self.nalfa,:self.nalfa]) 
                                      - np.einsum('pjjq->pq',self.h2e[:,:self.nalfa,:self.nalfa,:]))
        fao = self.mo_coeff.dot(self.ref_fock).dot(self.mo_coeff.T)
        fao = self.integrals.overlap_matrix().dot(fao).dot(self.integrals.overlap_matrix()) 
        # Energy for reference determinant
        self.eref = self.integrals.scalar_potential() + (np.einsum('ii',self.h1e[:self.nalfa,:self.nalfa] + self.ref_fock[:self.nalfa,:self.nalfa]))

        # Compute the Hamiltonian matrix
        self.ham = self.get_ham()

        # Core effective interaction
        self.V = (2 * np.einsum('pqkk->pq', self.h2e[:,:,:self.nalfa,:self.nalfa], optimize='optimal') 
                    - np.einsum('pkkq->pq', self.h2e[:,:self.nalfa,:self.nalfa,:], optimize='optimal'))
        
        self.sigma = self.ham.dot(self.mat_ci[:,0])
        

    def overlap(self, them):
        """Compute the many-body overlap with another CAS waveunction (them)"""
        return 0#esmf_coupling(self, them, self.ovlp, with_ref=self.with_ref)[0]

    def hamiltonian(self, them):
        """Compute the many-body Hamiltonian coupling with another CAS wavefunction (them)"""
        #eri = ao2mo.restore(1, self._scf._eri, self.mol.nao).reshape(self.mol.nao**2, self.mol.nao**2)
        return 0#esmf_coupling(self, them, self.ovlp, self.hcore, eri, self.enuc, with_ref=self.with_ref)

    def deallocate(self):
        # Reduce the memory footprint for storing 
        self._eri     = None
        self.h1e      = None
        self.h2e      = None
        self.ref_fock = None
        self.F_cas    = None
        self.ham      = None
    
    def get_ham(self):
        '''Build the full Hamiltonian in the CIS space'''
        ne   = self.nalfa
        nov  = self.nalfa * (self.nmo - self.nalfa)

        dij = np.identity(self.nalfa)
        dab = np.identity(self.nmo - self.nalfa)

        ham = np.zeros((self.ndet, self.ndet))
        if(self.with_ref):
            ham[0,0]   = self.eref
            ham[0,1:]  = np.sqrt(2) * np.reshape(self.ref_fock[:self.nalfa,self.nalfa:], (nov))
            ham[1:,0]  = np.sqrt(2) * np.reshape(self.ref_fock[:self.nalfa,self.nalfa:], (nov))

            hiajb = ( self.eref * np.einsum('ij,ab->iajb',dij,dab)
                     + np.einsum('ab,ij->iajb',self.ref_fock[ne:,ne:],dij)
                     - np.einsum('ij,ab->iajb',self.ref_fock[:ne,:ne],dab)
                     + 2 * np.einsum('aijb->iajb',self.h2e[ne:,:ne,:ne,ne:])
                         - np.einsum('abji->iajb',self.h2e[ne:,ne:,:ne,:ne]))
            ham[1:,1:] = np.reshape(np.reshape(hiajb,(ne,self.nmo-self.nalfa,-1)),(self.ndet-1,-1))
        else:
            hiajb = ( self.eref * np.einsum('ij,ab->iajb',dij,dab)
                     + np.einsum('ab,ij->iajb',self.ref_fock[ne:,ne:],dij)
                     - np.einsum('ij,ab->iajb',self.ref_fock[:ne,:ne],dab)
                     + 2 * np.einsum('aijb->iajb',self.h2e[ne:,:ne,:ne,ne:])
                         - np.einsum('abji->iajb',self.h2e[ne:,ne:,:ne,:ne]))
            ham[:,:] = np.reshape(np.reshape(hiajb,(ne,self.nmo-self.nalfa,-1)),(self.ndet,-1))
        return ham


    def get_rdm1(self, v1, v2, transition=False):                                                                              
        '''Compute the total 1RDM for the current state'''                                                                     
        ne = self.nalfa
        if(self.with_ref):
            c1_0   = v1[0]
            c2_0   = v2[0]
            t1     = 1/np.sqrt(2) * np.reshape(v1[1:], (self.nalfa, self.nmo - self.nalfa))
            t2     = 1/np.sqrt(2) * np.reshape(v2[1:], (self.nalfa, self.nmo - self.nalfa))
        else:
            c1_0   = 0.0
            c2_0   = 0.0
            t1     = 1/np.sqrt(2) * np.reshape(v1[:], (self.nalfa, self.nmo - self.nalfa))
            t2     = 1/np.sqrt(2) * np.reshape(v2[:], (self.nalfa, self.nmo - self.nalfa))

        # Derive temporary matrices        
        ttOcc = t1.dot(t2.T) # (tt)_{ij}
        ttVir = t2.T.dot(t1) # (tt)_{ab}

        # Compute the 1RDM
        dm1 = np.zeros((self.nmo,self.nmo))
        dm1[:self.nalfa,:self.nalfa] = - ttOcc
        dm1[:self.nalfa,self.nalfa:] = c2_0 * t1
        dm1[self.nalfa:,:self.nalfa] = c1_0 * t2.T
        dm1[self.nalfa:,self.nalfa:] = ttVir
        if(not transition): dm1[:self.nalfa,:self.nalfa] += np.identity(self.nanalfa)

        return 2 * dm1

    def get_rdm12(self):
        '''Compute the total 1RDM and 2RDM for the current state'''
        ne = self.nalfa
        if(self.with_ref):
            c0   = self.mat_ci[0,0]
            t    = 1/np.sqrt(2) * np.reshape(self.mat_ci[1:,0],(self.nalfa, self.nmo - self.nalfa))
        else:
            c0   = 0.0
            t    = 1/np.sqrt(2) * np.reshape(self.mat_ci[:,0],(self.nalfa, self.nmo - self.nalfa))
        kron = np.identity(self.nmo)

        # Derive temporary matrices        
        ttOcc = t.dot(t.T) # (tt)_{ij}
        ttVir = t.T.dot(t) # (tt)_{ab}

        # Compute the 1RDM
        dm1 = np.zeros((self.nmo,self.nmo))
        dm1[:self.nalfa,:self.nalfa] = (np.identity(self.nalfa) - ttOcc)
        dm1[:self.nalfa,self.nalfa:] = c0 * t
        dm1[self.nalfa:,:self.nalfa] = c0 * t.T
        dm1[self.nalfa:,self.nalfa:] = ttVir
        dm1 *= 2

        # Compute the 2RDM
        dm2 = np.zeros((self.nmo,self.nmo,self.nmo,self.nmo))
        # ijkl block
        dm2[:ne,:ne,:ne,:ne] += 4 * np.einsum('ij,kl->ijkl', kron[:ne,:ne], kron[:ne,:ne])
        dm2[:ne,:ne,:ne,:ne] -= 4 * np.einsum('ij,kl->ijkl', kron[:ne,:ne], ttOcc)
        dm2[:ne,:ne,:ne,:ne] -= 4 * np.einsum('ij,kl->ijkl', ttOcc, kron[:ne,:ne])
        dm2[:ne,:ne,:ne,:ne] -= 2 * np.einsum('il,kj->ijkl', kron[:ne,:ne], kron[:ne,:ne])
        dm2[:ne,:ne,:ne,:ne] += 2 * np.einsum('il,kj->ijkl', kron[:ne,:ne], ttOcc)
        dm2[:ne,:ne,:ne,:ne] += 2 * np.einsum('il,kj->ijkl', ttOcc, kron[:ne,:ne])
        # ijka block
        dm2[:ne,:ne,:ne,ne:] += 4 * c0 * np.einsum('ij,ka->ijka', kron[:ne,:ne], t)
        dm2[:ne,:ne,:ne,ne:] -= 2 * c0 * np.einsum('kj,ia->ijka', kron[:ne,:ne], t)
        dm2[:ne,ne:,:ne,:ne] = np.einsum('ijka->kaij',dm2[:ne,:ne,:ne,ne:])
        # ijak block
        dm2[:ne,:ne,ne:,:ne] += 4 * c0 * np.einsum('ij,ak->ijak', kron[:ne,:ne], t.T)
        dm2[:ne,:ne,ne:,:ne] -= 2 * c0 * np.einsum('ik,aj->ijak', kron[:ne,:ne], t.T)
        dm2[ne:,:ne,:ne,:ne] = np.einsum('ijak->akij',dm2[:ne,:ne,ne:,:ne])
        # ijab block
        dm2[:ne,:ne,ne:,ne:] += 4 * np.einsum('ij,ab->ijab', kron[:ne,:ne], ttVir)
        dm2[:ne,:ne,ne:,ne:] -= 2 * np.einsum('ib,ja->ijab', t, t)
        dm2[ne:,ne:,:ne,:ne] = np.einsum('ijab->abij', dm2[:ne,:ne,ne:,ne:])
        # iabj block
        dm2[:ne,ne:,ne:,:ne] += 4 * np.einsum('ia,jb->iabj', t, t)
        dm2[:ne,ne:,ne:,:ne] -= 2 * np.einsum('ij,ba->iabj', kron[:ne,:ne], ttVir)
        dm2[ne:,:ne,:ne,ne:] = np.einsum('iabj->bjia', dm2[:ne,ne:,ne:,:ne])

        return dm1, dm2

    
    def get_trdm12(self, v1, v2):
        '''Compute the total 1RDM and 2RDM for the current state'''
        ne = self.nalfa
        if(self.with_ref):
            c1_0   = v1[0]
            c2_0   = v2[0]
            t1     = 1/np.sqrt(2) * np.reshape(v1[1:], (self.nalfa, self.nmo - self.nalfa))
            t2     = 1/np.sqrt(2) * np.reshape(v2[1:], (self.nalfa, self.nmo - self.nalfa))
        else:
            c1_0   = 0.0
            c2_0   = 0.0
            t1     = 1/np.sqrt(2) * np.reshape(v1[:], (self.nalfa, self.nmo - self.nalfa))
            t2     = 1/np.sqrt(2) * np.reshape(v2[:], (self.nalfa, self.nmo - self.nalfa))
        kron = np.identity(self.nmo)

        # Derive temporary matrices        
        ttOcc = t1.dot(t2.T) # (tt)_{ij}
        ttVir = t2.T.dot(t1) # (tt)_{ab}

        # Compute the 1RDM
        dm1 = np.zeros((self.nmo,self.nmo))
        dm1[:self.nalfa,:self.nalfa] = - ttOcc
        dm1[:self.nalfa,self.nalfa:] = c2_0 * t1
        dm1[self.nalfa:,:self.nalfa] = c1_0 * t2.T
        dm1[self.nalfa:,self.nalfa:] = ttVir
        dm1 *= 2

        # Dompute the 2RDM
        dm2 = np.zeros((self.nmo,self.nmo,self.nmo,self.nmo))
        # ijkl block
        dm2[:ne,:ne,:ne,:ne] -= 4 * np.einsum('ij,kl->ijkl', kron[:ne,:ne], ttOcc, optimize="optimal")
        dm2[:ne,:ne,:ne,:ne] -= 4 * np.einsum('ij,kl->ijkl', ttOcc, kron[:ne,:ne], optimize="optimal")
        dm2[:ne,:ne,:ne,:ne] += 2 * np.einsum('il,kj->ijkl', kron[:ne,:ne], ttOcc, optimize="optimal")
        dm2[:ne,:ne,:ne,:ne] += 2 * np.einsum('il,kj->ijkl', ttOcc, kron[:ne,:ne], optimize="optimal")
        # ijka block
        dm2[:ne,:ne,:ne,ne:] += 4 * c2_0 * np.einsum('ij,ka->ijka', kron[:ne,:ne], t1, optimize="optimal")
        dm2[:ne,:ne,:ne,ne:] -= 2 * c2_0 * np.einsum('kj,ia->ijka', kron[:ne,:ne], t1, optimize="optimal")
        dm2[:ne,ne:,:ne,:ne] = np.einsum('ijka->kaij',dm2[:ne,:ne,:ne,ne:], optimize="optimal")
        # ijak block
        dm2[:ne,:ne,ne:,:ne] += 4 * c1_0 * np.einsum('ij,ak->ijak', kron[:ne,:ne], t2.T, optimize="optimal")
        dm2[:ne,:ne,ne:,:ne] -= 2 * c1_0 * np.einsum('ik,aj->ijak', kron[:ne,:ne], t2.T, optimize="optimal")
        dm2[ne:,:ne,:ne,:ne] = np.einsum('ijak->akij',dm2[:ne,:ne,ne:,:ne], optimize="optimal")
        # ijab block
        dm2[:ne,:ne,ne:,ne:] += 4 * np.einsum('ij,ab->ijab', kron[:ne,:ne], ttVir, optimize="optimal")
        dm2[:ne,:ne,ne:,ne:] -= 2 * np.einsum('ib,ja->ijab', t1, t2, optimize="optimal")
        dm2[ne:,ne:,:ne,:ne] = np.einsum('ijab->abij', dm2[:ne,:ne,ne:,ne:], optimize="optimal")
        # iabj block
        dm2[:ne,ne:,ne:,:ne] += 4 * np.einsum('ia,jb->iabj', t1, t2, optimize="optimal")
        dm2[:ne,ne:,ne:,:ne] -= 2 * np.einsum('ij,ba->iabj', kron[:ne,:ne], ttVir, optimize="optimal")
        dm2[ne:,:ne,:ne,ne:] = np.einsum('iabj->bjia', dm2[:ne,ne:,ne:,:ne], optimize="optimal")

        return dm1, dm2


    def canonicalize(self):
        """Rotate to canonical CI representation"""
        if(self.with_ref):
            t    = 1/np.sqrt(2) * np.reshape(self.mat_ci[1:,0],(self.nalfa, self.nmo - self.nalfa))
        else:
            t    = 1/np.sqrt(2) * np.reshape(self.mat_ci[:,0],(self.nalfa, self.nmo - self.nalfa))

        # Get SVD
        u, s, vt = np.linalg.svd(t)
        
        # Transform orbitals
        self.mo_coeff[:,:self.nalfa] = self.mo_coeff[:,:self.nalfa].dot(u)
        self.mo_coeff[:,self.nalfa:] = self.mo_coeff[:,self.nalfa:].dot(vt.T)

        for k in range(self.mat_ci.shape[1]):
            if(self.with_ref):
                tia = np.reshape(self.mat_ci[1:,k],(self.nalfa, self.nmo - self.nalfa))
                tia = np.linalg.multi_dot((u.T, tia, vt.T))
                self.mat_ci[1:,k] = np.reshape(tia,(self.nalfa * (self.nmo - self.nalfa)))
            else:
                tia    = np.reshape(self.mat_ci[:,k],(self.nalfa, self.nmo - self.nalfa))
                tia = np.linalg.multi_dot((u.T, tia, vt.T))
                self.mat_ci[:,k] = np.reshape(tia,(self.nalfa * (self.nmo - self.nalfa)))


    def get_gen_fock(self, v1, v2, transition=False):
        """Build generalised Fock matrix
           NOTE: This is a symmetrised version to minimise the number of einsums in 
                 the computation of transition generalised Fock matrices.
        """
        ne = self.nalfa
        if(self.with_ref):
            c1_0, c2_0 = v1[0], v2[0]
            t1     = 1/np.sqrt(2) * np.reshape(v1[1:], (self.nalfa, self.nmo - self.nalfa))
            t2     = 1/np.sqrt(2) * np.reshape(v2[1:], (self.nalfa, self.nmo - self.nalfa))
        else:
            c1_0, c2_0 = 0.0, 0.0
            t1     = 1/np.sqrt(2) * np.reshape(v1[:], (self.nalfa, self.nmo - self.nalfa))
            t2     = 1/np.sqrt(2) * np.reshape(v2[:], (self.nalfa, self.nmo - self.nalfa))

        if(transition): dij  = np.zeros((ne,ne))
        else: dij = np.identity(ne)
        
        # Compute the 1RDM
        gamma = np.zeros((self.nmo, self.nmo))
        gamma[:ne,:ne] = - t1.dot(t2.T)
        if(not transition): gamma[:ne,:ne] += np.identity(self.nalfa)
        gamma[ne:,ne:] = t2.T.dot(t1)
        if(self.with_ref):
            gamma[:ne,ne:] = c2_0 * t1
            gamma[ne:,:ne] = c1_0 * t2.T 
        # Symmetrise the density matrix
        gamma = 0.5 * (gamma + gamma.T)

        # Contributions to effective Fock
        Feff = (  2 * np.einsum('mnpq,qp->mn', self.h2e[:,:ne,:ne,:ne], gamma[:ne,:ne].T) 
                    - np.einsum('mqpn,pq->mn', self.h2e[:,:ne,:ne,:ne], gamma[:ne,:ne])   
                + 2 * np.einsum('mnpq,qp->mn', self.h2e[:,:ne,ne:,ne:], gamma[ne:,ne:].T) 
                    - np.einsum('mqpn,pq->mn', self.h2e[:,ne:,ne:,:ne], gamma[ne:,ne:]))   
        M = (  2 * np.einsum('mnia,ai->mn', self.h2e[:,:,:ne,ne:], t1.T) 
                 - np.einsum('main,ia->mn', self.h2e[:,ne:,:ne,:], t1))  
        N = (  2 * np.einsum('mnai,ia->mn', self.h2e[:,:,ne:,:ne], t2) 
                 - np.einsum('mian,ai->mn', self.h2e[:,:ne,ne:,:], t2.T))  
        if(self.with_ref):
            Feff += 0.5 * (M[:,:ne] * c2_0 + N[:ne,:].T * c1_0)
            Feff += 0.5 * (N[:,:ne] * c1_0 + M[:ne,:].T * c2_0)

        # Compute actual terms
        F = np.zeros((self.nmo,self.nmo))
        F += np.dot(gamma, self.h1e + self.V)
        F[:ne,:] += Feff[:,:ne].T 
        F[:ne,:] += 0.5 * (np.dot(t1, N[:,ne:].T) + np.dot(t2, M[ne:,:]))
        if(not transition): F[:ne,:] -= np.dot(dij, self.V[:ne,:])
        F[ne:,:] += 0.5 * (np.dot(t2.T, M[:,:ne].T) + np.dot(t1.T, N[:ne,:]))

        # We need to add a factor of 2 at the end!
        return 2 * F


    def get_hessianOrbCI(self):
        """Compute the orbital-CIS component of the energy Hessian"""
        # Initialise with zeros
        H_OCI = np.zeros((self.nmo,self.nmo,self.ndet-1))

        for k in range(1,self.ndet):
            # Get transition generalised Fock matrix
            F = 2 * self.get_gen_fock(self.mat_ci[:,k], self.mat_ci[:,0], True)
            # Save component
            H_OCI[:,:,k-1] = 2*(F.T - F)

        return H_OCI


    def get_hessianOrbOrb(self):
        """Compute the orbital-orbital component of the energy Hessian"""
        # Get number of electrons
        ne = self.nalfa
        nmo = self.nmo

        if(self.with_ref):
            c0 = self.mat_ci[0,0]
            t  = 1/np.sqrt(2) * np.reshape(self.mat_ci[1:,0], (self.nalfa, self.nmo - self.nalfa))
        else:
            c0 = 0.0
            t  = 1/np.sqrt(2) * np.reshape(self.mat_ci[:,0], (self.nalfa, self.nmo - self.nalfa))
        # Compute the 1RDM
        gamma = np.zeros((self.nmo, self.nmo))
        gamma[:ne,:ne] = np.identity(self.nalfa) - t.dot(t.T)
        gamma[ne:,ne:] = t.T.dot(t)
        if(self.with_ref):
            gamma[:ne,ne:] = c0 * t
            gamma[ne:,:ne] = c0 * t.T 

        # Get the generalised Fock matrix
        F = self.get_gen_fock(self.mat_ci[:,0], self.mat_ci[:,0]) 

        # iajb 
        Y = np.zeros((nmo,nmo,nmo,nmo))
        Y[:ne,ne:,:ne,ne:]  = np.einsum('ijmn,abmn->iajb', self.dm2[:ne,:ne,:,:], self.h2e[ne:,ne:,:,:])
        Y[:ne,ne:,:ne,ne:] += np.einsum('imjn,amnb->iajb', self.dm2[:ne,:,:ne,:], self.h2e[ne:,:,:,ne:])
        Y[:ne,ne:,:ne,ne:] += np.einsum('imnj,amnb->iajb', self.dm2[:ne,:,:,:ne], self.h2e[ne:,:,:,ne:])

        # aibj term
        Iilkj = self.h2e[:ne,:ne,:ne,:ne]
        Y[ne:,:ne,ne:,:ne]  = 2 * np.einsum('ab,ij->aibj',gamma[ne:,ne:], self.V[:ne,:ne])
        Y[ne:,:ne,ne:,:ne] += 2 * np.einsum('la,kb,ilkj->aibj', t, t, 2 * Iilkj - Iilkj.transpose(0,3,2,1))

        # Build intermediate N matrix
        N = (  2 * np.einsum('mnai,ia->mn', self.h2e[:,:,ne:,:ne], t) 
                 - np.einsum('mian,ai->mn', self.h2e[:,:ne,ne:,:], t.T))  
        # iabj term
        Iaipj = self.h2e[ne:,:ne,:,:ne]
        Iappj = self.h2e[ne:,:,:,:ne]
        Y[:ne,ne:,ne:,:ne]  = 2 * np.einsum('ib,aj->iabj', gamma[:ne,ne:], self.V[ne:,:ne])
        Y[:ne,ne:,ne:,:ne] += 2 * np.einsum('bp,aipj->iabj', 
                                gamma[ne:,:], 4 * Iaipj - Iaipj.transpose(0,3,2,1) - Iappj[:,:,:ne,:].transpose(0,2,1,3))
        Y[:ne,ne:,ne:,:ne] += 2 * np.einsum('ib,aj->iabj', t, N[ne:,:ne])
        Y[:ne,ne:,ne:,:ne] += 2 * np.einsum('ic,kb,ackj->iabj', 
                                t, t, 2*Iappj[:,ne:,:ne,:] - Iappj[:,:ne,ne:,:].transpose(0,2,1,3))
        del N
        # aijb are obtained from symmetry 
        Y[ne:,:ne,:ne,ne:] = Y[:ne,ne:,ne:,:ne].transpose(2,3,0,1)

        # Build last part of Eq 10.8.53
        tmp = np.zeros((nmo,nmo,nmo,nmo))

        # iajb
        tmp[:ne,ne:,:ne,ne:]   = 2 * np.einsum('ij,ab->iajb', self.dm1[:ne,:ne], self.h1e[ne:,ne:])
        for a in range(self.nalfa,self.nmo): tmp[:ne,a,:ne,a]  -= (F + F.T)[:ne,:ne]
        tmp[:ne,ne:,:ne,ne:]  += 2 * Y[:ne,ne:,:ne,ne:]
        # aibj
        tmp[ne:,:ne,ne:,:ne]   = 2 * np.einsum('ab,ij->aibj', self.dm1[ne:,ne:], self.h1e[:ne,:ne])
        for i in range(self.nalfa): tmp[ne:,i,ne:,i]  -= (F + F.T)[ne:,ne:]
        tmp[ne:,:ne,ne:,:ne]  += 2 * Y[ne:,:ne,ne:,:ne]
        # iabj
        tmp[:ne,ne:,ne:,:ne]   = 2 * np.einsum('ib,aj->iabj', self.dm1[:ne,ne:], self.h1e[ne:,:ne])
        tmp[:ne,ne:,ne:,:ne]  += 2 * Y[:ne,ne:,ne:,:ne]
        # aijb
        tmp[ne:,:ne,:ne,ne:]   = 2 * np.einsum('aj,ib->aijb', self.dm1[ne:,:ne], self.h1e[:ne,ne:])
        tmp[ne:,:ne,:ne,ne:]  += 2 * Y[ne:,:ne,:ne,ne:]
        del Y

        # Apply the permutation operators and return result
        return tmp - np.einsum('qprs->pqrs', tmp) - np.einsum('pqsr->pqrs', tmp) + np.einsum('qpsr->pqrs', tmp)


    def get_hessianCICI(self):
        """Compute the CIS-CIS component of the energy Hessian"""
        if(self.ndet > 1):
            e0 = np.einsum('i,ij,j', np.asarray(self.mat_ci)[:,0], self.ham, np.asarray(self.mat_ci)[:,0],optimize="optimal")
            return 2.0 * np.einsum('ki,kl,lj->ij', 
                    self.mat_ci[:,1:], self.ham - e0 * np.identity(self.ndet), self.mat_ci[:,1:],optimize="optimal")
        else: 
            return np.zeros((0,0))


    def uniq_var_indices(self, frozen):
        mask = np.zeros((self.nmo, self.nmo),dtype=bool)
        mask[self.nalfa:,:self.nalfa] = True    
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
                frozen = np.asarray(frozen)
                mask[frozen] = mask[:,frozen] = False
        return mask