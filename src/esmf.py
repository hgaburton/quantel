#!/usr/bin/python3
# Author: Antoine Marie, Hugh G. A. Burton

import numpy as np
import scipy.linalg
from functools import reduce
from pyscf import scf, fci, __config__, ao2mo, lib, mcscf, ci
from pyscf.mcscf import mc_ao2mo
from gnme.cas_noci import cas_proj
from utils import delta_kron, orthogonalise

class esmf():
    def __init__(self, mol):
        self.mol        = mol
        self.nelec      = mol.nelec
        self._scf       = scf.RHF(mol)
        self.verbose    = mol.verbose
        self.stdout     = mol.stdout
        self.max_memory = self._scf.max_memory
        # Get AO integrals 
        self.get_ao_integrals()
        self.norb       = self.hcore.shape[0]
        self.na         = self.nelec[0]
        self.nb         = self.nelec[1]
        assert(self.na == self.nb)

        # Get number of determinants
        self.nDet      = self.na * (self.norb - self.na) + 1

        # Save mapping indices for unique orbital rotations
        self.frozen     = None
        self.rot_idx    = self.uniq_var_indices(self.norb, self.frozen)
        self.nrot       = np.sum(self.rot_idx)

        # Dimensions of problem 
        self.dim        = self.nrot + self.nDet - 1

    def copy(self):
        # Return a copy of the current object
        newcas = esmf(self.mol)
        newcas.initialise(self.mo_coeff, self.mat_ci, integrals=False)
        return newcas
#
#    def overlap(self, them):
#        # Represent the alternative CAS state in the current CI space
#        vec2 = cas_proj(self, them, self.ovlp) 
#        # Compute the overlap and return
#        return np.dot(np.asarray(self.mat_ci)[:,0].conj(), vec2)
#
#    def sanity_check(self):
#        '''Need to be run at the start of the kernel to verify that the number of 
#           orbitals and electrons in the CAS are consistent with the system '''
#        assert self.ncas > 0
#        ncore = self.ncore
#        nvir = self.mo_coeff.shape[1] - ncore - self.ncas
#        assert ncore >= 0
#        assert nvir >= 0
#        assert ncore * 2 + sum(self.nelecas) == self.mol.nelectron
#        assert 0 <= self.nelecas[0] <= self.ncas
#        assert 0 <= self.nelecas[1] <= self.ncas
#        return self
#
#
    def get_ao_integrals(self):
        self.enuc       = self._scf.energy_nuc()
        self.v1e        = self.mol.intor('int1e_nuc')  # Nuclear repulsion matrix elements
        self.t1e        = self.mol.intor('int1e_kin')  # Kinetic energy matrix elements
        self.hcore      = self.t1e + self.v1e          # 1-electron matrix elements in the AO basis
        self.norb       = self.hcore.shape[0]
        self.ovlp       = self.mol.intor('int1e_ovlp') # Overlap matrix
        self._scf._eri  = self.mol.intor("int2e", aosym="s8") # Two electron integrals

    def initialise(self, mo_guess, ci_guess, integrals=True):
        # Save orbital coefficients
        mo_guess = orthogonalise(mo_guess, self.ovlp)
        self.mo_coeff = mo_guess
        self.nmo = self.mo_coeff.shape[1]
 
        # Save CI coefficients
        ci_guess = orthogonalise(ci_guess, np.identity(self.nDet)) 
        self.mat_ci = ci_guess

        # Initialise integrals
        if(integrals): self.update_integrals()

    def get_rdm12(self):
        '''Compute the total 1RDM and 2RDM for the current state'''
        ne = self.na
        c0   = self.mat_ci[0,0]
        t    = 1/np.sqrt(2) * np.reshape(self.mat_ci[1:,0],(self.na, self.nmo - self.na))
        kron = np.identity(self.nmo)
        dij  = np.identity(ne)
        dab  = np.identity(self.nmo-ne)

        # Derive temporary matrices        
        ttOcc = t.dot(t.T) # (tt)_{ij}
        ttVir = t.T.dot(t) # (tt)_{ab}

        # Compute the 1RDM
        dm1 = np.zeros((self.nmo,self.nmo))
        dm1[:self.na,:self.na] = (np.identity(self.na) - ttOcc)
        dm1[:self.na,self.na:] = c0 * t
        dm1[self.na:,:self.na] = c0 * t.T
        dm1[self.na:,self.na:] = ttVir
        dm1 *= 2

        # Dompute the 2RDM
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
        ne = self.na
        c1_0   = v1[0]
        c2_0   = v2[0]
        t1     = 1/np.sqrt(2) * np.reshape(v1[1:], (self.na, self.nmo - self.na))
        t2     = 1/np.sqrt(2) * np.reshape(v2[1:], (self.na, self.nmo - self.na))
        kron = np.identity(self.nmo)
        dij  = np.identity(ne)
        dab  = np.identity(self.nmo-ne)

        # Derive temporary matrices        
        ttOcc = t1.dot(t2.T) # (tt)_{ij}
        ttVir = t2.T.dot(t1) # (tt)_{ab}

        # Compute the 1RDM
        dm1 = np.zeros((self.nmo,self.nmo))
        dm1[:self.na,:self.na] = - ttOcc
        dm1[:self.na,self.na:] = c2_0 * t1
        dm1[self.na:,:self.na] = c1_0 * t2.T
        dm1[self.na:,self.na:] = ttVir
        dm1 *= 2

        # Dompute the 2RDM
        dm2 = np.zeros((self.nmo,self.nmo,self.nmo,self.nmo))
        # ijkl block
        #dm2[:ne,:ne,:ne,:ne] += 4 * np.einsum('ij,kl->ijkl', kron[:ne,:ne], kron[:ne,:ne])
        dm2[:ne,:ne,:ne,:ne] -= 4 * np.einsum('ij,kl->ijkl', kron[:ne,:ne], ttOcc)
        dm2[:ne,:ne,:ne,:ne] -= 4 * np.einsum('ij,kl->ijkl', ttOcc, kron[:ne,:ne])
        #dm2[:ne,:ne,:ne,:ne] -= 2 * np.einsum('il,kj->ijkl', kron[:ne,:ne], kron[:ne,:ne])
        dm2[:ne,:ne,:ne,:ne] += 2 * np.einsum('il,kj->ijkl', kron[:ne,:ne], ttOcc)
        dm2[:ne,:ne,:ne,:ne] += 2 * np.einsum('il,kj->ijkl', ttOcc, kron[:ne,:ne])
        # ijka block
        dm2[:ne,:ne,:ne,ne:] += 4 * c2_0 * np.einsum('ij,ka->ijka', kron[:ne,:ne], t1)
        dm2[:ne,:ne,:ne,ne:] -= 2 * c2_0 * np.einsum('kj,ia->ijka', kron[:ne,:ne], t1)
        dm2[:ne,ne:,:ne,:ne] = np.einsum('ijka->kaij',dm2[:ne,:ne,:ne,ne:])
        # ijak block
        dm2[:ne,:ne,ne:,:ne] += 4 * c1_0 * np.einsum('ij,ak->ijak', kron[:ne,:ne], t2.T)
        dm2[:ne,:ne,ne:,:ne] -= 2 * c1_0 * np.einsum('ik,aj->ijak', kron[:ne,:ne], t2.T)
        dm2[ne:,:ne,:ne,:ne] = np.einsum('ijak->akij',dm2[:ne,:ne,ne:,:ne])
        # ijab block
        dm2[:ne,:ne,ne:,ne:] += 4 * np.einsum('ij,ab->ijab', kron[:ne,:ne], ttVir)
        dm2[:ne,:ne,ne:,ne:] -= 2 * np.einsum('ib,ja->ijab', t1, t2)
        dm2[ne:,ne:,:ne,:ne] = np.einsum('ijab->abij', dm2[:ne,:ne,ne:,ne:])
        # iabj block
        dm2[:ne,ne:,ne:,:ne] += 4 * np.einsum('ia,jb->iabj', t1, t2)
        dm2[:ne,ne:,ne:,:ne] -= 2 * np.einsum('ij,ba->iabj', kron[:ne,:ne], ttVir)
        dm2[ne:,:ne,:ne,ne:] = np.einsum('iabj->bjia', dm2[:ne,ne:,ne:,:ne])

        return dm1, dm2

#    def deallocate(self):
#        # Reduce the memory footprint for storing 
#        self._eri   = None
#        self.ppoo   = None
#        self.popo   = None
#        self.h1e    = None
#        self.h1eff  = None
#        self.h2eff  = None
#        self.F_core = None
#        self.F_cas  = None
#        self.ham    = None
#
    @property
    def energy(self):
        ''' Compute the energy corresponding to a given set of
             one-el integrals, two-el integrals, 1- and 2-RDM '''
        E  = self.enuc
        E += np.einsum('pq,pq', self.h1e, self.dm1, optimize="optimal")
        E += 0.5 * np.einsum('pqrs,pqrs', self.h2e, self.dm2, optimize="optimal")
        return E
#
#    @property
#    def s2(self):
#        ''' Compute the spin of a given FCI vector '''
#        return self.fcisolver.spin_square(self.mat_ci[:,0], self.ncas, self.nelecas)[0]
#
    @property
    def gradient(self):
        g_orb = self.get_orbital_gradient()
        g_ci  = self.get_ci_gradient()  
        # Unpack matrices/vectors accordingly
        return np.concatenate((g_orb, g_ci))

    @property
    def hessian(self):
        ''' This method concatenate the orb-orb, orb-CI and CI-CI part of the Hessian '''
        H_OrbOrb = (self.get_hessianOrbOrb()[:,:,self.rot_idx])[self.rot_idx,:]
        H_CICI   = self.get_hessianCICI()
        H_OrbCI  = self.get_hessianOrbCI()[self.rot_idx,:]

        return np.block([[H_OrbOrb, H_OrbCI],
                         [H_OrbCI.T, H_CICI]])

#    def get_hessian_index(self, tol=1e-16):
#        eigs = scipy.linalg.eigvalsh(self.hessian)
#        ndown = 0
#        nzero = 0
#        nuphl = 0
#        for i in eigs:
#            if i < -tol:  ndown += 1
#            elif i > tol: nuphl +=1
#            else:         nzero +=1 
#        return ndown, nzero, nuphl
#
#
#    def pushoff(self, n, angle=np.pi/2, evec=None):
#        """Perturb along n Hessian directions"""
#        if evec is None:
#            eigval, eigvec = np.linalg.eigh(self.hessian)
#        else:
#            eigvec = evec
#        step = sum(eigvec[:,i] * angle for i in range(n))
#        self.take_step(step)
#
#    
#    def guess_casci(self, n):
#        self.mat_ci = np.linalg.eigh(self.ham)[1]
#        self.mat_ci[:,[0,n]] = self.mat_ci[:,[n,0]]
#
    def update_integrals(self):
        # One-electron Hamiltonian
        self.h1e = np.einsum('ip,ij,jq->pq', self.mo_coeff, self.hcore, self.mo_coeff,optimize="optimal")

        # Occupied orbitals
        self.h2e = ao2mo.incore.general(self._scf._eri, (self.mo_coeff, self.mo_coeff, self.mo_coeff, self.mo_coeff), compact=False)
        self.h2e = np.reshape(self.h2e, (self.nmo, self.nmo, self.nmo, self.nmo))

        # Reduced density matrices 
        self.dm1, self.dm2 = self.get_rdm12()

        # Fock matrix for reference determinant
        self.ref_fock = self.h1e + (2 * np.einsum('pqjj->pq',self.h2e[:,:,:self.na,:self.na]) 
                                      - np.einsum('pjjq->pq',self.h2e[:,:self.na,:self.na,:]))
        fao = self.mo_coeff.dot(self.ref_fock).dot(self.mo_coeff.T)
        fao = self.ovlp.dot(fao).dot(self.ovlp) 
        # Energy for reference determinant
        self.eref = self.enuc + (np.einsum('ii',self.h1e[:self.na,:self.na] + self.ref_fock[:self.na,:self.na]))

        self.ham = self.get_ham()

    def get_ham(self):
        '''Build the full Hamiltonian in the CIS space'''
        ne   = self.na
        nvir = self.nmo - self.na
        nov  = self.na * (self.nmo - self.na)

        dij = np.identity(self.na)
        dab = np.identity(self.nmo - self.na)

        ham = np.zeros((self.nDet, self.nDet))
        ham[0,0]   = self.eref
        ham[0,1:]  = np.sqrt(2) * np.reshape(self.ref_fock[:self.na,self.na:], (nov))
        ham[1:,0]  = np.sqrt(2) * np.reshape(self.ref_fock[:self.na,self.na:], (nov))

        hiajb = ( self.eref * np.einsum('ij,ab->iajb',dij,dab) 
                 + np.einsum('ab,ij->iajb',self.ref_fock[ne:,ne:],dij)
                 - np.einsum('ij,ab->iajb',self.ref_fock[:ne,:ne],dab) 
                 + 2 * np.einsum('aijb->iajb',self.h2e[ne:,:ne,:ne,ne:]) 
                     - np.einsum('abji->iajb',self.h2e[ne:,ne:,:ne,:ne])) 
        ham[1:,1:] = np.reshape(np.reshape(hiajb,(ne,self.nmo-self.na,-1)),(self.nDet-1,-1))
        return ham

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
        self.update_integrals()

    def rotate_orb(self,step): 
        '''Rotate the molecular orbital coefficients'''
        # Transform the step into correct structure
        orb_step = np.zeros((self.norb,self.norb))
        orb_step[self.rot_idx] = step
        self.mo_coeff = np.dot(self.mo_coeff, scipy.linalg.expm(orb_step - orb_step.T))


    def rotate_ci(self,step): 
        S       = np.zeros((self.nDet,self.nDet))
        S[1:,0] = step
        #self.mat_ci = np.dot(scipy.linalg.expm(S - S.T),self.mat_ci)
        self.mat_ci = np.dot(self.mat_ci, scipy.linalg.expm(S - S.T))
#
#    def get_h1eff(self):
#        '''CAS sapce one-electron hamiltonian
#
#        Returns:
#            A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
#            the second is the electronic energy from core.
#        '''
#        ncas  = self.ncas
#        ncore = self.ncore
#        nocc  = self.ncore + self.ncas
#
#        # Get core and active orbital coefficients 
#        mo_core = self.mo_coeff[:,:ncore]
#        mo_cas  = self.mo_coeff[:,ncore:ncore+ncas]
#
#        # Core density matrix (in AO basis)
#        self.core_dm = np.dot(mo_core, mo_core.T) * 2
#
#        # Core energy
#        energy_core  = self.enuc
#        energy_core += np.einsum('ij,ji', self.core_dm, self.hcore,optimize="optimal")
#        energy_core += self.vhf_c[:ncore,:ncore].trace()
#
#        # Get effective Hamiltonian in CAS space
#        h1eff  = np.einsum('ki,kl,lj->ij', mo_cas.conj(), self.hcore, mo_cas,optimize="optimal")
#        h1eff += self.vhf_c[ncore:nocc,ncore:nocc]
#        return h1eff, np.asscalar(energy_core)
#
#
#    def get_spin_dm1(self):
#        # Reformat the CI vector
#        civec = np.reshape(self.mat_ci[:,0],(self.nDeta,self.nDetb))
#        
#        # Compute spin density in the active space
#        alpha_dm1_cas, beta_dm1_cas = self.fcisolver.make_rdm1s(civec, self.ncas, self.nelecas)
#        spin_dens_cas = alpha_dm1_cas - beta_dm1_cas
#
#        # Transform into the AO basis
#        mo_cas = self.mo_coeff[:, self.ncore:self.ncore+self.ncas]
#        ao_spin_dens  = np.dot(mo_cas, np.dot(spin_dens_cas, mo_cas.T))
#        return ao_spin_dens
#
    def get_gen_fock(self, dm1, dm2, transition=False):
        """Build generalised Fock matrix"""
        # Initialise matrix
        F = np.zeros((self.nmo,self.nmo))

        # One-body contribution
        F += np.einsum('mq,nq->mn', dm1, self.h1e, optimize='optimal')

        # Two-body contribution
        F += np.einsum('mqrs,nqrs->mn', dm2, self.h2e, optimize='optimal')

        return F

    def get_orbital_gradient(self):
        ''' This method builds the orbital part of the gradient '''
        g_orb = 2 * self.get_gen_fock(self.dm1, self.dm2, False)
        return (g_orb.T - g_orb)[self.rot_idx]


    def get_ci_gradient(self):
        if(self.nDet > 1):
            return 2.0 * np.einsum('i,ij,jk->k', np.asarray(self.mat_ci)[:,0], self.ham, self.mat_ci[:,1:],optimize="optimal")
        else:
            return np.zeros((0))

    def get_old_ci_gradient(self):
        sigma = self.ham.dot(self.mat_ci)[:,0]
        vec   = self.mat_ci[:,0]
        return 2 * (sigma[1:] * vec[0] - sigma[0] * vec[1:])
       
    
    def get_hessianOrbCI(self):
        '''This method build the orb-CI part of the hessian'''
        H_OCI = np.zeros((self.norb,self.norb,self.nDet-1))
        for k in range(1,self.nDet):

            # Get transition density matrices
            dm1_ok, dm2_ok = self.get_trdm12(self.mat_ci[:,0], self.mat_ci[:,k])
            # Get transition density matrices
            dm1_ko, dm2_ko = self.get_trdm12(self.mat_ci[:,k], self.mat_ci[:,0])
            # Get transition generalised Fock matrix
            F = self.get_gen_fock(dm1_ok + dm1_ko, dm2_ok + dm2_ko)

            # Save component
            H_OCI[:,:,k-1] = 2*(F.T - F)

        return H_OCI


    def get_hessianOrbOrb(self):
        ''' This method build the orb-orb part of the hessian '''
        F = self.get_gen_fock(self.dm1, self.dm2) 
        dqs = np.identity(self.nmo)

        # Build Y intermediate from Helgaker Eq. 10.8.50
        Y  = np.einsum('prmn,qsmn->pqrs', self.dm2, self.h2e)
        Y += np.einsum('pmrn,qmns->pqrs', self.dm2, self.h2e)
        Y += np.einsum('pmnr,qmns->pqrs', self.dm2, self.h2e)

        # Build last part of Eq 10.8.53
        tmp   = 2 * np.einsum('pr,qs->pqrs', self.dm1, self.h1e)
        tmp  -=  np.einsum('pr,qs->pqrs',F + F.T, dqs) 
        tmp  += 2 * Y

        # Apply the permutation operators and return result
        return tmp - np.einsum('qprs->pqrs', tmp) - np.einsum('pqsr->pqrs', tmp) + np.einsum('qpsr->pqrs', tmp)

    def get_hessianCICI(self):
        ''' This method build the CI-CI part of the hessian '''
        if(self.nDet > 1):
            e0 = np.einsum('i,ij,j', np.asarray(self.mat_ci)[:,0], self.ham, np.asarray(self.mat_ci)[:,0],optimize="optimal")
            return 2.0 * np.einsum('ki,kl,lj->ij', 
                    self.mat_ci[:,1:], self.ham - e0 * np.identity(self.nDet), self.mat_ci[:,1:],optimize="optimal")
        else: 
            return np.zeros((0,0))


    def get_front_hessianCICI(self):
        ''' This method build the CI-CI part of the hessian '''
        sigma = self.ham.dot(self.mat_ci)[:,0]
        vec   = self.mat_ci[:,0]
        c0 = vec[0]

        tmp  = 2 * c0 * c0 * self.ham[1:,1:]
        tmp -= 2 * c0 * np.einsum('i,j->ij', vec[1:], self.ham[0,1:])
        tmp -= 2 * c0 * np.einsum('j,i->ij', vec[1:], self.ham[0,1:])
        tmp += 2 * self.ham[0,0] * np.einsum('i,j->ij', vec[1:], vec[1:])
        tmp -= 1 * np.einsum('i,j->ij', sigma[1:], vec[1:])
        tmp -= 1 * np.einsum('j,i->ij', sigma[1:], vec[1:])
        tmp -= 2 *  c0 * sigma[0] * np.identity(self.nDet-1)
        return tmp


#    def _eig(self, h, *args):
#        return scf.hf.eig(h, None)
#    def get_hcore(self, mol=None):
#        return self.hcore
#    def get_fock(self, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None):
#        return mcscf.casci.get_fock(self, mo_coeff, ci, eris, casdm1, verbose)
#    def cas_natorb(self, mo_coeff=None, ci=None, eris=None, sort=False, casdm1=None, verbose=None, with_meta_lowdin=True):
#        test = mcscf.casci.cas_natorb(self, mo_coeff, ci, eris, sort, casdm1, verbose, True)
#        return test
#    def canonicalize_(self):
#        # Compute canonicalised natural orbitals
#        ao2mo_level = getattr(__config__, 'mcscf_mc1step_CASSCF_ao2mo_level', 2)
#        self.mo_coeff, ci, self.mo_energy = mcscf.casci.canonicalize(
#                      self, self.mo_coeff, ci=self.mat_ci[:,0], 
#                      eris=mc_ao2mo._ERIS(self, self.mo_coeff, method='incore', level=ao2mo_level),
#                      sort=True, cas_natorb=True, casdm1=self.dm1_cas)
#
#        # Insert new "occupied" ci vector
#        self.mat_ci[:,0] = ci.ravel()
#        self.mat_ci = orthogonalise(self.mat_ci, np.identity(self.nDet))
#
#        # Update integrals
#        self.update_integrals()
#        return

    def uniq_var_indices(self, nmo, frozen):
        ''' This function creates a matrix of boolean of size (norb,norb). 
            A True element means that this rotation should be taken into 
            account during the optimization. Taken from pySCF.mcscf.casscf '''
        mask = np.zeros((self.norb,self.norb),dtype=bool)
        mask[self.na:,:self.na] = True    # Active-Core rotations
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
                frozen = np.asarray(frozen)
                mask[frozen] = mask[:,frozen] = False
        return mask

#    def CASRDM1_to_RDM1(self, dm1_cas, transition=False):
#        ''' Transform 1-RDM from CAS space into full MO space'''
#        ncore = self.ncore
#        ncas = self.ncas
#        dm1 = np.zeros((self.norb,self.norb))
#        if transition is False and self.ncore > 0:
#            dm1[:ncore,:ncore] = 2 * np.identity(self.ncore, dtype="int") 
#        dm1[ncore:ncore+ncas,ncore:ncore+ncas] = dm1_cas
#        return dm1
#
#    def CASRDM2_to_RDM2(self, dm1_cas, dm2_cas, transition=False):
#        ''' This method takes a 2-RDM in the CAS space and transform it to the full MO space '''
#        ncore = self.ncore
#        ncas = self.ncas
#        norb = self.norb
#        nocc = ncore + ncas
#
#        dm1 = self.CASRDM1_to_RDM1(dm1_cas,transition)
#        dm2 = np.zeros((norb,norb,norb,norb))
#        if transition is False:
#            # Core contributions
#            for i in range(ncore):
#                for j in range(ncore):
#                    for k in range(ncore):
#                        for l in range(ncore):
#                            dm2[i,j,k,l]  = 4 * delta_kron(i,j) * delta_kron(k,l) 
#                            dm2[i,j,k,l] -= 2 * delta_kron(i,l) * delta_kron(k,j)
#
#                    for p in range(ncore,nocc):
#                        for q in range(ncore,nocc):
#                            dm2[i,j,p,q] = 2 * delta_kron(i,j) * dm1[q,p]
#                            dm2[p,q,i,j] = dm2[i,j,p,q]
#
#                            dm2[i,q,p,j] = - delta_kron(i,j) * dm1[q,p]
#                            dm2[p,j,i,q] = dm2[i,q,p,j]
#
#        else:
#            for i in range(ncore):
#                for j in range(ncore):
#                    for p in range(ncore,ncore+ncas):
#                        for q in range(ncore,ncore+ncas):
#                            dm2[i,j,p,q] = 2 * delta_kron(i,j) * dm1[q,p]
#                            dm2[p,q,i,j] = dm2[i,j,p,q]
#
#                            dm2[i,q,p,j] = - delta_kron(i,j) * dm1[q,p]
#                            dm2[p,j,i,q] = dm2[i,q,p,j]
#
#        # Insert the active-active sector
#        dm2[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc] = dm2_cas 
#        return dm2


    def get_numerical_gradient(self,eps=1e-3):
        grad = np.zeros((self.dim))
        for i in range(self.dim):
            x1 = np.zeros(self.dim)
            x2 = np.zeros(self.dim)
                
            x1[i] += eps
            x2[i] -= eps
                
            self.take_step(x1)
            E1 = self.energy
            self.restore_last_step()

            self.take_step(x2)
            E2 = self.energy
            self.restore_last_step()

            grad[i] = (E1 - E2) / (2 * eps)

        return grad

    def get_numerical_hessian(self,eps=1e-3):
        Hess = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(i,self.dim):
                x1 = np.zeros(self.dim)
                x2 = np.zeros(self.dim)
                x3 = np.zeros(self.dim)
                x4 = np.zeros(self.dim)
                
                x1[i] += eps; x1[j] += eps
                x2[i] += eps; x2[j] -= eps
                x3[i] -= eps; x3[j] += eps
                x4[i] -= eps; x4[j] -= eps
                
                self.take_step(x1)
                E1 = self.energy
                self.restore_last_step()

                self.take_step(x2)
                E2 = self.energy
                self.restore_last_step()

                self.take_step(x3)
                E3 = self.energy
                self.restore_last_step()

                self.take_step(x4)
                E4 = self.energy
                self.restore_last_step()

                Hess[i,j] = ((E1 - E2) - (E3 - E4)) / (4 * eps * eps)
                if(i!=j): Hess[j,i] = Hess[i,j]

        return Hess


##### Main #####
if __name__ == '__main__':
    import sys, re, os
    from opt.eigenvector_following import EigenFollow
    from pyscf import gto

    np.set_printoptions(linewidth=10000)

    def read_config(file):
        f = open(file,"r")
        lines = f.read().splitlines()
        basis, charge, spin, frozen, cas, grid_option, Hind, maxit = 'sto-3g', 0, 0, 0, (0,0), 1000, None, 1000
        for line in lines:
            if re.match('basis', line) is not None:
                basis = str(re.split(r'\s', line)[-1])
            elif re.match('charge', line) is not None:
                charge = int(re.split(r'\s', line)[-1])
            elif re.match('spin', line) is not None:
                spin = int(re.split(r'\s', line)[-1])
            elif re.match('frozen', line) is not None:
                frozen = int(re.split(r'\s', line)[-1])
            elif re.match('seed', line) is not None:
                np.random.seed(int(line.split()[-1]))
            elif re.match('index', line) is not None:
                Hind = int(line.split()[-1])
            elif re.match('maxit', line) is not None:
                maxit = int(line.split()[-1])
            elif re.match('cas', line) is not None:
                tmp = list(re.split(r'\s', line)[-1])
                cas = (int(tmp[1]), int(tmp[3]))
            elif re.match('grid', line) is not None:
                if re.split(r'\s', line)[-1] == 'full':
                    grid_option = re.split(r'\s', line)[-1]
                else:
                    grid_option = int(re.split(r'\s', line)[-1])
        return basis, charge, spin, frozen, cas, grid_option, Hind, maxit

    mol = gto.Mole(symmetry=False,unit='A')
    mol.atom = sys.argv[1]
    basis, charge, spin, frozen, cas, grid_option, Hind, maxit = read_config(sys.argv[2])
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()
    myhf = mol.RHF().run()

    # Initialise CAS object
    myesmf = esmf(mol)

    myhf.mo_coeff[:,[0,-1]] = myhf.mo_coeff[:,[0,-1]].dot(scipy.linalg.expm(np.matrix([[0,0.2],[-0.2,0]])))
    print(myhf.mo_occ)
    print("Actual reference energy", myhf.energy_tot())
    

    np.random.seed(7)
    for nsample in range(10):
        del myesmf
        myesmf = esmf(mol)
        # Set orbital coefficients
        np.set_printoptions(precision=4,suppress=True)
        ci_guess = np.random.rand(myesmf.nDet,myesmf.nDet)

        myesmf.initialise(myhf.mo_coeff, ci_guess)
        #print("Energy = ", myesmf.energy)
        #hess = myesmf.hessian
        #num_hess = myesmf.get_numerical_hessian()
        #grad = myesmf.gradient
        #num_grad = myesmf.get_numerical_gradient()
        #print("Analytic Hessian = ")
        #print(hess)
        #print(num_hess)
        #print("Analytic gradient = ")
        #print(grad)
        #print(num_grad)
        #quit()
        #print(myhf.energy_tot())
#    quit()
        opt = EigenFollow(minstep=0.0,rtrust=0.15)
        opt.run(myesmf, thresh=1e-8, maxit=500, index=0)
        #mycas.canonicalize_()

    print()
    print("  Final energy = {: 16.10f}".format(myesmf.energy))
#    print("         <S^2> = {: 16.10f}".format(mycas.s2))

    print()
    print("  Canonical natural orbitals:      ")
    print("  ---------------------------------")
    print(" {:^5s}  {:^10s}  {:^10s}".format("  Orb","Occ.","Energy"))
    print("  ---------------------------------")
    for i in range(mycas.ncore + mycas.ncas):
        print(" {:5d}  {: 10.6f}  {: 10.6f}".format(i+1, mycas.mo_occ[i], mycas.mo_energy[i]))
    print("  ---------------------------------")
