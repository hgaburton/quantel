#!/usr/bin/python3
# Author: Antoine Marie, Hugh G. A. Burton

import numpy as np
import scipy.linalg
from functools import reduce
from pyscf import scf, __config__, ao2mo
from xesla.utils.linalg import delta_kron, orthogonalise
from .wavefunction import Wavefunction
from xesla.gnme.pcid_noci import pcid_coupling

class PCID(Wavefunction):
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

    @property
    def dim(self):
        return self.nrot + self.nDet - 1


    def copy(self):
        # Return a copy of the current object
        newcas = PCID(self.mol)
        newcas.initialise(self.mo_coeff, self.mat_ci, integrals=False)
        return newcas


    def overlap(self, them):
        return pcid_coupling(self, them, self.ovlp)[0]


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

    def get_rdm1(self):
        '''Compute the total 1RDM for the current state'''
        ne = self.na
        c0   = self.mat_ci[0,0]
        t    = np.reshape(self.mat_ci[1:,0],(self.na, self.nmo - self.na))
        kron = np.identity(self.nmo)
        dij  = np.identity(ne)
        dab  = np.identity(self.nmo-ne)

        # Derive temporary matrices        
        ttOcc = t.dot(t.T) # (tt)_{ij}
        ttVir = t.T.dot(t) # (tt)_{ab}

        # Compute the 1RDM
        dm1 = np.zeros((self.nmo,self.nmo))
        dm1[:self.na,:self.na] = (np.identity(self.na) - ttOcc)
        dm1[self.na:,self.na:] = ttVir
        return 2*dm1

    def get_rdm12(self):
        '''Compute the total 1RDM and 2RDM for the current state'''
        ne = self.na
        c0   = self.mat_ci[0,0]
        t    = np.reshape(self.mat_ci[1:,0],(self.na, self.nmo - self.na))
        kron = np.identity(self.nmo)
        dij  = np.identity(ne)
        dab  = np.identity(self.nmo-ne)

        # Derive temporary matrices        
        ttOcc = t.dot(t.T) # (tt)_{ij}
        ttVir = t.T.dot(t) # (tt)_{ab}

        # Compute the 1RDM
        dm1 = np.zeros((self.nmo,self.nmo))
        dm1[:self.na,:self.na] = (np.identity(self.na) - np.diag(np.diag(ttOcc)))
        dm1[self.na:,self.na:] = np.diag(np.diag(ttVir))
        dm1 *= 2

        # Compute the 2RDM
        dm2 = np.zeros((self.nmo,self.nmo,self.nmo,self.nmo))
        # ijkl block
        dm2[:ne,:ne,:ne,:ne] += 4 * np.einsum('ij,kl->ijkl', kron[:ne,:ne], kron[:ne,:ne]) #
        dm2[:ne,:ne,:ne,:ne] -= 2 * np.einsum('il,kj->ijkl', kron[:ne,:ne], kron[:ne,:ne]) #
        dm2[:ne,:ne,:ne,:ne] -= 4 * np.einsum('ij,kl->ijkl', kron[:ne,:ne], np.diag(np.diag(ttOcc))) #
        dm2[:ne,:ne,:ne,:ne] += 2 * np.einsum('il,kj->ijkl', kron[:ne,:ne], np.diag(np.diag(ttOcc))) #
        dm2[:ne,:ne,:ne,:ne] -= 4 * np.einsum('ij,kl->ijkl', np.diag(np.diag(ttOcc)), kron[:ne,:ne]) #
        dm2[:ne,:ne,:ne,:ne] += 2 * np.einsum('il,kj->ijkl', np.diag(np.diag(ttOcc)), kron[:ne,:ne]) #
        dm2[:ne,:ne,:ne,:ne] += 2 * np.einsum('ik,jl,ji->ijkl', kron[:ne,:ne], kron[:ne,:ne], ttOcc) #
        # ijka block = 0
        # ijak block = 0
        # ijab block
        dm2[:ne,:ne,ne:,ne:] += 4 * np.einsum('ij,ab->ijab',    kron[:ne,:ne], np.diag(np.diag(ttVir))) #
        dm2[:ne,:ne,ne:,ne:] -= 4 * np.einsum('ij,ja,ab->ijab', kron[:ne,:ne], np.power(t,2), kron[ne:,ne:]) #
        dm2[ne:,ne:,:ne,:ne]  =     np.einsum('ijab->abij', dm2[:ne,:ne,ne:,ne:]) #
        # iabj block
        dm2[:ne,ne:,ne:,:ne] += 2 * np.einsum('ij,ja,ab->iabj', kron[:ne,:ne], np.power(t,2), kron[ne:,ne:]) #
        dm2[:ne,ne:,ne:,:ne] -= 2 * np.einsum('ij,ab->iabj',    kron[:ne,:ne], np.diag(np.diag(ttVir))) #
        dm2[ne:,:ne,:ne,ne:]  =     np.einsum('iabj->bjia', dm2[:ne,ne:,ne:,:ne]) #
        # iajb block
        dm2[:ne,ne:,:ne,ne:] += 2 * np.einsum('ij,ja,ab->iajb',kron[:ne,:ne], t, kron[ne:,ne:]) * c0 #
        # aibj block
        dm2[ne:,:ne,ne:,:ne] += 2 * np.einsum('ij,ja,ab->aibj',kron[:ne,:ne], t, kron[ne:,ne:]) * c0 #
        # abcd block
        dm2[ne:,ne:,ne:,ne:] += 2 * np.einsum('ac,bd,da->abcd', kron[ne:,ne:], kron[ne:,ne:], ttVir) # 

        return dm1, dm2

    def get_trdm12(self, v1, v2):
        '''Compute the total 1RDM and 2RDM for the current state'''
        ne = self.na
        c1_0   = v1[0]
        c2_0   = v2[0]
        t1     = np.reshape(v1[1:], (self.na, self.nmo - self.na))
        t2     = np.reshape(v2[1:], (self.na, self.nmo - self.na))
        kron = np.identity(self.nmo)
        dij  = np.identity(ne)
        dab  = np.identity(self.nmo-ne)

        # Derive temporary matrices        
        ttOcc = t1.dot(t2.T) # (tt)_{ij}
        ttVir = t2.T.dot(t1) # (tt)_{ab}

        # Compute the 1RDM
        dm1 = np.zeros((self.nmo,self.nmo))
        dm1[:self.na,:self.na] = - np.diag(np.diag(ttOcc))
        dm1[self.na:,self.na:] = np.diag(np.diag(ttVir))
        dm1 *= 2

        # Dompute the 2RDM
        dm2 = np.zeros((self.nmo,self.nmo,self.nmo,self.nmo))
        # ijkl block
        dm2[:ne,:ne,:ne,:ne] -= 4 * np.einsum('ij,kl->ijkl', kron[:ne,:ne], np.diag(np.diag(ttOcc))) #
        dm2[:ne,:ne,:ne,:ne] += 2 * np.einsum('il,kj->ijkl', kron[:ne,:ne], np.diag(np.diag(ttOcc))) #
        dm2[:ne,:ne,:ne,:ne] -= 4 * np.einsum('ij,kl->ijkl', np.diag(np.diag(ttOcc)), kron[:ne,:ne]) #
        dm2[:ne,:ne,:ne,:ne] += 2 * np.einsum('il,kj->ijkl', np.diag(np.diag(ttOcc)), kron[:ne,:ne]) #
        dm2[:ne,:ne,:ne,:ne] += 2 * np.einsum('ik,jl,ji->ijkl', kron[:ne,:ne], kron[:ne,:ne], ttOcc) #
        # ijka block = 0
        # ijak block = 0
        # ijab block
        dm2[:ne,:ne,ne:,ne:] += 4 * np.einsum('ij,ab->ijab',    kron[:ne,:ne], np.diag(np.diag(ttVir))) #
        dm2[:ne,:ne,ne:,ne:] -= 4 * np.einsum('ij,ja,ab->ijab', kron[:ne,:ne], np.multiply(t1,t2), kron[ne:,ne:]) #
        dm2[ne:,ne:,:ne,:ne]  =     np.einsum('ijab->abij', dm2[:ne,:ne,ne:,ne:]) #
        # iabj block
        dm2[:ne,ne:,ne:,:ne] += 2 * np.einsum('ij,ja,ab->iabj', kron[:ne,:ne], np.multiply(t1,t2), kron[ne:,ne:]) #
        dm2[:ne,ne:,ne:,:ne] -= 2 * np.einsum('ij,ab->iabj',    kron[:ne,:ne], np.diag(np.diag(ttVir))) #
        dm2[ne:,:ne,:ne,ne:]  =     np.einsum('iabj->bjia', dm2[:ne,ne:,ne:,:ne]) #
        # iajb block
        dm2[:ne,ne:,:ne,ne:] += 2 * np.einsum('ij,ja,ab->iajb',kron[:ne,:ne], t2, kron[ne:,ne:]) * c1_0 #
        # aibj block
        dm2[ne:,:ne,ne:,:ne] += 2 * np.einsum('ij,ja,ab->aibj',kron[:ne,:ne], t1, kron[ne:,ne:]) * c2_0 #
        # abcd block
        dm2[ne:,ne:,ne:,ne:] += 2 * np.einsum('ac,bd,da->abcd', kron[ne:,ne:], kron[ne:,ne:], ttVir) # 

        return dm1, dm2

    def deallocate(self):
        # Reduce the memory footprint for storing 
        self.h1e      = None
        self.h2e      = None
        self.ref_fock = None
        self.ham      = None

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

    def get_hessian_index(self, tol=1e-16):
        eigs = scipy.linalg.eigvalsh(self.hessian)
        ndown = 0
        nzero = 0
        nuphl = 0
        for i in eigs:
            if i < -tol:  ndown += 1
            elif i > tol: nuphl +=1
            else:         nzero +=1 
        return ndown, nzero, nuphl

#    def guess_casci(self, n):
#        self.mat_ci = np.linalg.eigh(self.ham)[1]
#        self.mat_ci[:,[0,n]] = self.mat_ci[:,[n,0]]
#
    def update_integrals(self):
        # One-electron Hamiltonian
        self.h1e = np.einsum('ip,ij,jq->pq', 
                             self.mo_coeff, self.hcore, self.mo_coeff,optimize="optimal")

        # Occupied orbitals
        self.h2e = ao2mo.incore.general(self._scf._eri, 
                                       (self.mo_coeff, self.mo_coeff, self.mo_coeff, self.mo_coeff), 
                                       compact=False)
        self.h2e = np.reshape(self.h2e, (self.nmo, self.nmo, self.nmo, self.nmo))

        # Reduced density matrices 
        self.dm1, self.dm2 = self.get_rdm12()

        # Fock matrix for reference determinant
        self.ref_fock = self.h1e + (2 * np.einsum('pqjj->pq',self.h2e[:,:,:self.na,:self.na]) 
                                      - np.einsum('pjjq->pq',self.h2e[:,:self.na,:self.na,:]))
        fao = self.mo_coeff.dot(self.ref_fock).dot(self.mo_coeff.T)
        fao = self.ovlp.dot(fao).dot(self.ovlp) 
        # Energy for reference determinant
        self.eref = self.enuc + (np.einsum('ii',self.h1e[:self.na,:self.na] 
                                         + self.ref_fock[:self.na,:self.na]))

        self.ham = self.get_ham()

    def get_ham(self):
        '''Build the full Hamiltonian in the pCID space'''
        ne   = self.na
        nvir = self.nmo - self.na
        nov  = self.na * (self.nmo - self.na)

        dij = np.identity(self.na)
        dab = np.identity(self.nmo - self.na)

        ham = np.zeros((self.nDet, self.nDet))
        ham[0,0]   = self.eref
        ham[0,1:]  = np.reshape(np.einsum('iaia->ia',self.h2e[:ne,ne:,:ne,ne:]), (nov))
        ham[1:,0]  = ham[0,1:]

        hiajb = ( self.eref * np.einsum('ij,ab->iajb',dij,dab) 
                 + 2 * np.einsum('ab,ij->iajb',np.diag(np.diag(self.ref_fock[ne:,ne:])),dij)
                 - 2 * np.einsum('ij,ab->iajb',np.diag(np.diag(self.ref_fock[:ne,:ne])),dab) 
                 + np.einsum('abab,ij->iajb', self.h2e[ne:,ne:,ne:,ne:], dij) 
                 + np.einsum('ab,jiji->iajb', dab, self.h2e[:ne,:ne,:ne,:ne])
                 - 4 * np.einsum('ij,ab,abji->iajb',dij,dab,self.h2e[ne:,ne:,:ne,:ne])
                 + 2 * np.einsum('ij,ab,aijb->iajb',dij,dab,self.h2e[ne:,:ne,:ne,ne:])
                ) 
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
#        print((g_orb.T - g_orb))
        return (g_orb.T - g_orb)[self.rot_idx]


    def get_ci_gradient(self):
        if(self.nDet > 1):
            return 2.0 * np.einsum('i,ij,jk->k', np.asarray(self.mat_ci)[:,0], self.ham, self.mat_ci[:,1:],optimize="optimal")
        else:
            return np.zeros((0))

    
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
        for i in  range(self.norb):
            for j in range(i):
                mask[i,j] = True
        #mask[self.na:,:self.na] = True    # Active-Core rotations
        #if frozen is not None:
        #    if isinstance(frozen, (int, np.integer)):
        #        mask[:frozen] = mask[:,:frozen] = False
        #    else:
        #        frozen = np.asarray(frozen)
        #        mask[frozen] = mask[:,frozen] = False
        return mask

    def save_to_disk(self,tag):
        """Save object to disk with prefix 'tag'"""
        # Get the Hessian index
        hindices = self.get_hessian_index()

        # Save coefficients, CI, and energy
        np.savetxt(tag+'.mo_coeff', self.mo_coeff, fmt="% 20.16f")
        np.savetxt(tag+'.mat_ci',   self.mat_ci, fmt="% 20.16f")
        np.savetxt(tag+'.energy',   
                   np.array([[self.energy, hindices[0], hindices[1], 0]]), 
                   fmt="% 18.12f % 5d % 5d % 12.6f")


    def read_from_disk(self,tag):
        """Read object from disk with prefix 'tag'"""
        # Read MO coefficient and CI coefficients
        mo_coeff = np.genfromtxt(tag+".mo_coeff")
        ci_coeff = np.genfromtxt(tag+".mat_ci")

        # Initialise object
        self.initialise(mo_coeff, ci_coeff)

    def overlap(self, them):
        """Compute the many-body overlap with another CAS waveunction (them)"""
        return 0# esmf_coupling(self, them, self.ovlp, with_ref=self.with_ref)[0]

    def hamiltonian(self, them):
        """Compute the many-body Hamiltonian coupling with another CAS wavefunction (them)"""
        eri = ao2mo.restore(1, self._scf._eri, self.mol.nao).reshape(self.mol.nao**2, self.mol.nao**2)
        return 0#esmf_coupling(self, them, self.ovlp, self.hcore, eri, self.enuc, with_ref=self.with_ref)
