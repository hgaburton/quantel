#!/usr/bin/env python
# Author: Antoine Marie, Hugh G. A. Burton

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import sys, re
import numpy as np
import scipy.linalg
from pyscf import gto, scf, ao2mo, fci, mcscf
from newton_raphson import NewtonRaphson


def delta_kron(i,j):
    if i==j: return 1
    else: return 0

def orthogonalise(mat, metric, thresh=1e-12):
    ortho = np.einsum('ki,kl,lj',mat.conj(), metric, mat)
    ortho_test = np.linalg.norm(ortho - np.identity(ortho.shape[0]))
    if ortho_test > thresh:
        s, u = np.linalg.eigh(ortho)
        print(s)
        mat = mat.dot(u.dot(np.diag(np.power(s,-0.5))))
    return mat

class ss_casscf():
    def __init__(self, mol, ncas, nelecas, ncore=None):
        self.mol        = mol
        self.nelec      = mol.nelec
        self._scf       = scf.RHF(mol)
        self.verbose    = mol.verbose
        self.stdout     = mol.stdout
        self.max_memory = myhf.max_memory
        self.ncas       = ncas                        # Number of active orbitals
        if isinstance(nelecas, (int, np.integer)):
            nelecb = (nelecas-mol.spin)//2
            neleca = nelecas - nelecb
            self.nelecas = (neleca, nelecb)     # Tuple of number of active electrons
        else:
            self.nelecas = np.asarray((nelecas[0],nelecas[1])).astype(int)

        if ncore is None:
            ncorelec = self.mol.nelectron - sum(self.nelecas)
            assert ncorelec % 2 == 0
            assert ncorelec >= 0
            self.ncore = ncorelec // 2
        else:
            self.ncore = ncore

        # Get AO integrals 
        self.get_ao_integrals()

        # Get number of determinants
        self.nDeta      = (scipy.special.comb(self.ncas,self.nelecas[0])).astype(int)
        self.nDetb      = (scipy.special.comb(self.ncas,self.nelecas[1])).astype(int)
        self.nDet       = (self.nDeta*self.nDetb).astype(int)

        # Save mapping indices for unique orbital rotations
        self.frozen     = None
        self.rot_idx    = self.uniq_var_indices(self.norb, self.frozen)
        self.nrot       = np.sum(self.rot_idx)

        # Define the FCI solver
        self.fcisolver = fci.direct_spin1.FCISolver(mol)

    def sanity_check(self):
        ''' Need to be run at the start of the kernel to verify that the number of orbitals and electrons in the CAS are consistent with the system '''
        assert self.ncas > 0
        ncore = self.ncore
        nvir = self.mo_coeff.shape[1] - ncore - self.ncas
        assert ncore >= 0
        assert nvir >= 0
        assert ncore * 2 + sum(self.nelecas) == self.mol.nelectron
        assert 0 <= self.nelecas[0] <= self.ncas
        assert 0 <= self.nelecas[1] <= self.ncas
        return self


    def get_ao_integrals(self):
        self.enuc       = self._scf.energy_nuc()
        self.v1e        = self.mol.intor('int1e_nuc')       # Nuclear repulsion matrix elements
        self.t1e        = self.mol.intor('int1e_kin')       # Kinetic energy matrix elements
        self.hcore      = self.t1e + self.v1e              # 1-electron matrix elements in the AO basis
        self.norb       = self.hcore.shape[0]
        self.ovlp       = self.mol.intor('int1e_ovlp')      # Overlpa matrix
       

    def initialise(self, mo_guess, ci_guess):
        # Save orbital coefficients
        mo_guess = orthogonalise(mo_guess, self.ovlp)
        self.mo_coeff = mo_guess
 
        # Save CI coefficients
        ci_guess = orthogonalise(ci_guess, np.identity(self.nDet)) 
        self.mat_CI = ci_guess

        # Initialise integrals
        self.update_integrals()
        self.energy       = self.get_energy()
        self.s2, self.mul = self.spin_square()


    def get_energy(self):
        ''' Compute the energy corresponding to a given set of one-el integrals, two-el integrals, 1- and 2-RDM '''
        dm1 = self.CASRDM1_to_RDM1(self.dm1_cas)
        dm2 = self.CASRDM2_to_RDM2(self.dm1_cas, self.dm2_cas)
        return np.einsum('pq,pq', self.h1e, dm1) + np.einsum('pqrs,pqrs', self.eri, dm2) + self.enuc 

    def spin_square(self):
        ''' Compute the spin of a given FCI vector '''
        return self.fcisolver.spin_square(self.mat_CI[:,0], self.ncas, self.nelecas)


    def update_integrals(self):
        # One-electron Hamiltonian
        self.h1e = np.einsum('ip,ij,jq->pq', self.mo_coeff, self.hcore, self.mo_coeff)

        # Two-electron integrals
        self.eri = np.asarray(self.mol.ao2mo(self.mo_coeff))
        self.eri = ao2mo.restore(1, self.eri, self.norb)

        # Effective Hamiltonians in CAS space
        self.h1eff, self.energy_core = self.get_h1eff()
        self.h2eff = self.get_h2eff()
        self.h2eff = ao2mo.restore(1, self.h2eff, self.ncas)

        # Reduced density matrices 
        self.dm1_cas, self.dm2_cas = self.get_casrdm_12()

        # Fock matrices
        self.F_core, self.F_cas = self.get_fock_matrices()

        # Hamiltonian in active space
        self.ham = self.fcisolver.pspace(self.h1eff, self.h2eff, self.ncas, self.nelecas, np=1000000)[1]

    def restore_last_step(self):
        self.mo_coeff = self.mo_coeff_save.copy()
        self.mat_CI   = self.mat_CI_save.copy()

    def save_last_step(self):
        self.mo_coeff_save = self.mo_coeff.copy()
        self.mat_CI_save   = self.mat_CI.copy()

    def take_step(self,step):
        # Save our last position
        self.save_last_step()

        # Take steps in orbital and CI space
        self.rotate_orb(step[:self.nrot])
        self.rotate_ci(step[self.nrot:])

        # Finally, update our integrals for the new coefficients
        self.update_integrals()
        self.energy       = self.get_energy()
        self.s2, self.mul = self.spin_square()

    def rotate_orb(self,step): 
        orb_step = np.zeros((self.norb,self.norb))
        orb_step[self.rot_idx] = step
        self.mo_coeff = np.dot(self.mo_coeff, scipy.linalg.expm(orb_step - orb_step.T))

    def rotate_ci(self,step): 
        S       = np.zeros((self.nDet,self.nDet))
        S[1:,0] = step
        self.mat_CI = np.dot(self.mat_CI, scipy.linalg.expm(S - S.T))

    def get_h1eff(self):
        '''CAS sapce one-electron hamiltonian

        Returns:
            A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
            the second is the electronic energy from core.
        '''
        # Get core and active orbital coefficients 
        mo_core = self.mo_coeff[:,:self.ncore]
        mo_cas  = self.mo_coeff[:,self.ncore:self.ncore+self.ncas]

        energy_core = self.enuc
        if mo_core.size == 0:
            corevhf = 0
        else:
            core_dm = 2.0 * np.einsum('ik,jk->ij', mo_core, mo_core.conj())
            vj, vk = self._scf.get_jk(self.mol, core_dm, 1, True, True, None)
            corevhf = vj - 0.5 * vk
            energy_core = np.einsum('ij,ji', core_dm, self.hcore + 0.5 * corevhf).real

        # Get effective Hamiltonian in CAS space
        h1eff = np.einsum('ki,kl,lj->ij',mo_cas.conj(), self.hcore + corevhf, mo_cas)
        return h1eff, energy_core


    def get_h2eff(self):
        '''Compute the active space two-particle Hamiltonian. '''
        nocc = self.ncore + self.ncas
        if self._scf._eri is not None:
            return ao2mo.full(self._scf._eri, self.mo_coeff[:,self.ncore:nocc], max_memory=self.max_memory)
        else:
            return ao2mo.full(self.mol, self.mo_coeff[:,self.ncore:nocc], verbose=self.verbose, max_memory=self.max_memory)


    def get_casrdm_12(self):
        civec = np.reshape(self.mat_CI[:,0],(self.nDeta,self.nDetb))
        dm1_cas, dm2_cas = self.fcisolver.make_rdm12(civec, self.ncas, self.nelecas)
        return dm1_cas, 0.5 * dm2_cas


    def get_fock_matrices(self):
        ''' Compute the core part of the generalized Fock matrix '''
        ncore = self.ncore
        nocc  = self.ncore + self.ncas
        # Core contribution
        Fcore = self.h1e + 2*np.einsum('pqii->pq', self.eri[:, :, :ncore, :ncore]) - np.einsum('piiq->pq', self.eri[:, :ncore, :ncore, :])
        # Active space contribution
        Fcas  = (  1.0 * np.einsum('tu,pqtu->pq', self.dm1_cas, self.eri[:, :, ncore:nocc, ncore:nocc]) 
                 - 0.5 * np.einsum('tu,puqt->pq', self.dm1_cas, self.eri[:, ncore:nocc, :, ncore:nocc]))
        return Fcore, Fcas


    def get_orbital_gradient(self):
        ''' This method builds the orbital part of the gradient '''
        g_orb = np.zeros((self.norb,self.norb))
        ncore = self.ncore
        ncas  = self.ncas
        nocc  = ncore + ncas
        nvir  = self.norb - nocc

        #virtual-core rotations g_{ai}
        if ncore > 0 and nvir > 0:
            g_orb[nocc:,:ncore] = 4 * (self.F_core + self.F_cas)[nocc:,:ncore]
        #active-core rotations g_{ti}
        if ncore > 0:
            g_orb[ncore:nocc,:ncore]  = 4 * (self.F_core + self.F_cas)[ncore:nocc,:ncore] 
            g_orb[ncore:nocc,:ncore] -= 2 * np.einsum('tv,iv->ti', self.dm1_cas, self.F_core[:ncore,ncore:nocc]) 
            g_orb[ncore:nocc,:ncore] -= 4 * np.einsum('tvxy,ivxy->ti', self.dm2_cas, self.eri[:ncore,ncore:nocc,ncore:nocc,ncore:nocc])
        #virtual-active rotations g_{at}
        if nvir > 0:
            g_orb[nocc:,ncore:nocc]  = 2 * np.einsum('tv,av->at', self.dm1_cas, self.F_core[nocc:,ncore:nocc])
            g_orb[nocc:,ncore:nocc] += 4 * np.einsum('tvxy,avxy->at', self.dm2_cas, self.eri[nocc:,ncore:nocc,ncore:nocc,ncore:nocc])

        return (g_orb - g_orb.T)[self.rot_idx]


    def get_ci_gradient(self):
        ''' This method build the CI part of the gradient '''
        # Return gradient
        return 2.0 * np.einsum('i,ij,jk->k', self.mat_CI[:,0], self.ham, self.mat_CI[:,1:])
    
    def get_gradient(self):
        g_orb = self.get_orbital_gradient()
        g_ci  = self.get_ci_gradient()  

        # Unpack matrices/vectors accordingly
        return np.concatenate((g_orb, g_ci))

    def CASRDM1_to_RDM1(self, dm1_cas, transition=False):
        ''' Transform 1-RDM from CAS space into full MO space'''
        ncore = self.ncore
        ncas = self.ncas
        dm1 = np.zeros((self.norb,self.norb))
        if transition is False and self.ncore > 0:
            dm1[:ncore,:ncore] = 2 * np.identity(self.ncore, dtype="int") 
        dm1[ncore:ncore+ncas,ncore:ncore+ncas] = dm1_cas
        return dm1

    def CASRDM2_to_RDM2(self, dm1_cas, dm2_cas, transition=False):
        ''' This method takes a 2-RDM in the CAS space and transform it to the full MO space '''
        ncore = self.ncore
        ncas = self.ncas
        norb = self.norb
        nocc = ncore + ncas

        dm2 = np.zeros((norb,norb,norb,norb))
        if transition is False:
            dm1 = self.CASRDM1_to_RDM1(dm1_cas)
            for i in range(ncore):
                for j in range(ncore):
                    for p in range(ncore,ncore+ncas):
                        for q in range(ncore,ncore+ncas):
                            dm2[i,j,p,q] = delta_kron(i,j) * dm1[p,q] - delta_kron(i,q) * delta_kron(j,p)
                            dm2[p,q,i,j] = dm2[i,j,p,q]

                            dm2[p,i,j,q] = 2 * delta_kron(i,p) * delta_kron(j,q) - 0.5 * delta_kron(i,j) * dm1[p,q]
                            dm2[j,q,p,i] = dm2[p,i,j,q]

                    for k in range(ncore):
                        for l in range(ncore):
                            dm2[i,j,k,l] = 2 * delta_kron(i,j) * delta_kron(k,l) - delta_kron(i,l) * delta_kron(j,k)

        else:
            dm1 = self.CASRDM1_to_RDM1(dm1_cas,True)
            for i in range(ncore):
                for j in range(ncore):
                    for p in range(ncore,ncore+ncas):
                        for q in range(ncore,ncore+ncas):
                            dm2[i,j,p,q] = delta_kron(i,j) * dm1[p,q]
                            dm2[p,q,i,j] = delta_kron(i,j) * dm1[q,p]

                            dm2[p,i,j,q] = -0.5 * delta_kron(i,j) * dm1[p,q]
                            dm2[j,q,p,i] = -0.5 * delta_kron(i,j) * dm1[q,p]

        dm2[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc] = dm2_cas # Finally we add the uvxy sector
        return dm2

    def get_tCASRDM12(self,ci1,ci2):
        ''' This method compute the 1- and 2-electrons transition density matrix between the ci vectors ci1 and ci2 '''
        ncas = self.ncas
        nelecas = self.nelecas
        if len(ci1.shape)==1:
            nDeta = scipy.special.comb(ncas,nelecas[0]).astype(int)
            nDetb = scipy.special.comb(ncas,nelecas[1]).astype(int)
            ci1 = ci1.reshape((nDeta,nDetb))
        if len(ci2.shape)==1:
            nDeta = scipy.special.comb(ncas,nelecas[0]).astype(int)
            nDetb = scipy.special.comb(ncas,nelecas[1]).astype(int)
            ci2 = ci2.reshape((nDeta,nDetb))

        t_dm1_cas, t_dm2_cas = self.fcisolver.trans_rdm12(ci1,ci2,ncas,nelecas)

        #t_dm2_cas = t_dm2_cas + np.einsum('pqrs->qpsr',t_dm2_cas) #TODO check this

        return t_dm1_cas.T, 0.5*t_dm2_cas

    def get_ham_commutator(self):
        ''' This method build the Hamiltonian commutator matrices '''
        ncore = self.ncore; ncas = self.ncas; norb = self.norb
        nocc = ncore + ncas; nvir = norb - nocc

        # Initialise output
        H_ai = np.zeros((nvir,ncore,self.nDet,self.nDet))
        H_at = np.zeros((nvir,ncas, self.nDet,self.nDet))
        H_ti = np.zeros((ncas,ncore,self.nDet,self.nDet))

        # Compute contribution for each CI contribution
        mat_id = np.identity(self.nDet)
        for i in range(self.nDet):
            for j in range(self.nDet):
                dm1_cas, dm2_cas = self.get_tCASRDM12(mat_id[i],mat_id[j])
                dm1 = self.CASRDM1_to_RDM1(dm1_cas,True)
                dm2 = self.CASRDM2_to_RDM2(dm1_cas,dm2_cas,True)
                if ncore>0 and nvir>0:
                    one_el = ( np.einsum('pa,pi->ai', self.h1e[:,nocc:], dm1[:,:ncore]) 
                             + np.einsum('ap,ip->ai', self.h1e[nocc:,:], dm1[:ncore,:]) )
                    two_el = ( np.einsum('pars,pirs->ai', self.eri[:,nocc:,:,:], dm2[:,:ncore,:,:]) 
                             + np.einsum('aqrs,iqrs->ai', self.eri[nocc:,:,:,:], dm2[:ncore,:,:,:]) 
                             + np.einsum('pqra,pqri->ai', self.eri[:,:,:,nocc:], dm2[:,:,:,:ncore]) 
                             + np.einsum('pqas,pqis->ai', self.eri[:,:,nocc:,:], dm2[:,:,:ncore,:]) )
                    H_ai[:,:,i,j] = one_el + two_el

                if nvir>0:
                    one_el = ( np.einsum('pa,pt->at', self.h1e[:,nocc:], dm1[:,ncore:nocc]) 
                             + np.einsum('ap,tp->at', self.h1e[nocc:,:], dm1[ncore:nocc,:]) )
                    two_el = ( np.einsum('pars,ptrs->at', self.eri[:,nocc:,:,:], dm2[:,ncore:nocc,:,:])  
                             + np.einsum('aqrs,tqrs->at', self.eri[nocc:,:,:,:], dm2[ncore:nocc,:,:,:]) 
                             + np.einsum('pqra,pqrt->at', self.eri[:,:,:,nocc:], dm2[:,:,:,ncore:nocc]) 
                             + np.einsum('pqas,pqts->at', self.eri[:,:,nocc:,:], dm2[:,:,ncore:nocc,:]) )
                    H_at[:,:,i,j] = one_el + two_el

                if ncore>0:
                    one_el = ( np.einsum('pt,pi->ti', self.h1e[:,ncore:nocc], dm1[:,:ncore]) 
                             - np.einsum('pi,pt->ti', self.h1e[:,:ncore], dm1[:,ncore:nocc]) 
                             + np.einsum('tp,ip->ti', self.h1e[ncore:nocc,:], dm1[:ncore,:]) 
                             - np.einsum('ip,tp->ti', self.h1e[:ncore,:], dm1[ncore:nocc,:]) )
                    two_el = ( np.einsum('ptrs,pirs->ti', self.eri[:,ncore:nocc,:,:], dm2[:,:ncore,:,:]) 
                             - np.einsum('pirs,ptrs->ti', self.eri[:,:ncore,:,:], dm2[:,ncore:nocc,:,:]) 
                             + np.einsum('tqrs,iqrs->ti', self.eri[ncore:nocc,:,:,:], dm2[:ncore,:,:,:]) 
                             - np.einsum('iqrs,tqrs->ti', self.eri[:ncore,:,:,:], dm2[ncore:nocc,:,:,:]) 
                             + np.einsum('pqrt,pqri->ti', self.eri[:,:,:,ncore:nocc], dm2[:,:,:,:ncore]) 
                             - np.einsum('pqri,pqrt->ti', self.eri[:,:,:,:ncore], dm2[:,:,:,ncore:nocc]) 
                             + np.einsum('pqts,pqis->ti', self.eri[:,:,ncore:nocc,:], dm2[:,:,:ncore,:]) 
                             - np.einsum('pqis,pqts->ti', self.eri[:,:,:ncore,:], dm2[:,:,ncore:nocc,:]) )
                    H_ti[:,:,i,j] = one_el + two_el

        return H_ai, H_at, H_ti


    def get_hessianOrbCI(self):
        ''' This method build the orb-CI part of the hessian '''
        ncore = self.ncore; ncas = self.ncas; nocc = ncore + ncas
        nvir = self.norb - nocc

        H_OCI = np.zeros((self.norb,self.norb,self.nDet-1))
        mat_CI = self.mat_CI

        H_ai, H_at, H_ti = self.get_ham_commutator()

        ci0 = mat_CI[:,0]

        h1e = self.h1e
        eri = self.eri

        for k in range(len(mat_CI)-1): # Loop on Hessian indices
                cleft = mat_CI[:,k+1]
                if ncore>0 and nvir>0:
                    H_OCI[nocc:, :ncore, k] = 2*np.einsum('k,aikl,l->ai', cleft, H_ai, ci0)
                if nvir>0:
                    H_OCI[nocc:, ncore:nocc, k] = 2*np.einsum('k,aikl,l->ai', cleft, H_at, ci0)
                if ncore>0:
                    H_OCI[ncore:nocc, :ncore, k] = 2*np.einsum('k,aikl,l->ai', cleft, H_ti, ci0)

        H_OCI = H_OCI + np.einsum('pqs->qps',H_OCI)

        return H_OCI


    def get_hessianOrbOrb(self):
        ''' This method build the orb-orb part of the hessian '''
        norb = self.norb; ncore = self.ncore; ncas = self.ncas
        nocc = ncore + ncas; nvir = norb - nocc

        Htmp = np.zeros((norb,norb,norb,norb))
        F_tot = self.F_core + self.F_cas

        # Temporary identity matrices 
        id_cor = np.identity(ncore)
        id_vir = np.identity(nvir)
        id_cas = np.identity(ncas)

        #virtual-core virtual-core H_{ai,bj}
        if ncore>0 and nvir>0:
            tmp  = self.eri[nocc:,:ncore,nocc:,:ncore]

            aibj = self.eri[nocc:,:ncore,nocc:,:ncore]
            abij = self.eri[nocc:,nocc:,:ncore,:ncore]
            ajbi = self.eri[nocc:,:ncore,nocc:,:ncore]

            abij = np.einsum('abij->aibj', abij)
            ajbi = np.einsum('ajbi->aibj', ajbi)

            Htmp[nocc:,:ncore,nocc:,:ncore] = ( 4 * (4 * aibj - abij - ajbi)  
                                              + 4 * np.einsum('ij,ab->aibj', id_cor, F_tot[nocc:,nocc:]) 
                                              - 4 * np.einsum('ab,ij->aibj', id_vir, F_tot[:ncore,:ncore]) )

        #virtual-core virtual-active H_{ai,bt}
        if ncore>0 and nvir>0:
            aibv = self.eri[nocc:,:ncore,nocc:,ncore:nocc]
            avbi = self.eri[nocc:,ncore:nocc,nocc:,:ncore]
            abvi = self.eri[nocc:,nocc:,ncore:nocc,:ncore]

            avbi = np.einsum('avbi->aibv', avbi)
            abvi = np.einsum('abvi->aibv', abvi)

            Htmp[nocc:,:ncore,nocc:,ncore:nocc] = ( 2 * np.einsum('tv,aibv->aibt', self.dm1_cas, 4 * aibv - avbi - abvi) 
                                                  - 2 * np.einsum('ab,tvxy,vixy ->aibt', id_vir, self.dm2_cas, self.eri[ncore:nocc, :ncore, ncore:nocc, ncore:nocc]) 
                                                  - 2 * np.einsum('ab,ti->aibt', id_vir, F_tot[ncore:nocc, :ncore]) 
                                                  - 1 * np.einsum('ab,tv,vi->aibt', id_vir, self.dm1_cas, self.F_core[ncore:nocc, :ncore]) )

        #virtual-active virtual-core H_{bt,ai}
        if ncore>0 and nvir>0:
             Htmp[nocc:, ncore:nocc, nocc:, :ncore] = np.einsum('aibt->btai', Htmp[nocc:, :ncore, nocc:, ncore:nocc])

        #virtual-core active-core H_{ai,tj}
        if ncore>0 and nvir>0:
            aivj = self.eri[nocc:,:ncore,ncore:nocc,:ncore]
            avji = self.eri[nocc:,ncore:nocc,:ncore,:ncore]
            ajvi = self.eri[nocc:,:ncore,ncore:nocc,:ncore]

            avji = np.einsum('avji->aivj', avji)
            ajvi = np.einsum('ajvi->aivj', ajvi)

            Htmp[nocc:,:ncore,ncore:nocc,:ncore] = ( 2 * np.einsum('tv,aivj->aitj', (2 * id_cas - self.dm1_cas), 4 * aivj - avji - ajvi) 
                                                   - 2 * np.einsum('ji,tvxy,avxy -> aitj', id_cor, self.dm2_cas, self.eri[nocc:,ncore:nocc,ncore:nocc,ncore:nocc]) 
                                                   + 4 * np.einsum('ij,at-> aitj', id_cor, F_tot[nocc:, ncore:nocc]) 
                                                   - 1 * np.einsum('ij,tv,av-> aitj', id_cor, self.dm1_cas, self.F_core[nocc:, ncore:nocc]) )

        #active-core virtual-core H_{tj,ai}
        if ncore>0 and nvir>0:
            Htmp[ncore:nocc, :ncore, nocc:, :ncore] = np.einsum('aitj->tjai',Htmp[nocc:,:ncore,ncore:nocc,:ncore])

        #virtual-active virtual-active H_{at,bu}
        if nvir>0:
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc]  = ( 4 * np.einsum('tuvx,abvx->atbu', self.dm2_cas, self.eri[nocc:,nocc:,ncore:nocc,ncore:nocc]) 
                                                          + 4 * np.einsum('txvu,axbv->atbu', self.dm2_cas, self.eri[nocc:,ncore:nocc,nocc:,ncore:nocc]) 
                                                          + 4 * np.einsum('txuv,axbv->atbu', self.dm2_cas, self.eri[nocc:,ncore:nocc,nocc:,ncore:nocc]) )
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] -= ( 2 * np.einsum('ab,tvxy,uvxy->atbu', id_vir, self.dm2_cas, self.eri[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc]) 
                                                          + 1 * np.einsum('ab,tv,uv->atbu', id_vir, self.dm1_cas, self.F_core[ncore:nocc,ncore:nocc]) )
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] -= ( 2 * np.einsum('ab,uvxy,tvxy->atbu', id_vir, self.dm2_cas, self.eri[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc]) 
                                                          + 1 * np.einsum('ab,uv,tv->atbu', id_vir, self.dm1_cas, self.F_core[ncore:nocc,ncore:nocc]) )
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] +=   2 * np.einsum('tu,ab->atbu', self.dm1_cas, self.F_core[nocc:, nocc:])

        #active-core virtual-active H_{ti,au}
        if ncore>0 and nvir>0:
            avti = self.eri[nocc:, ncore:nocc, ncore:nocc, :ncore]
            aitv = self.eri[nocc:, :ncore, ncore:nocc, ncore:nocc]
            atvi = self.eri[nocc:, ncore:nocc, ncore:nocc, :ncore]
            aitv = np.einsum('aitv->avti', aitv)
            atvi = np.einsum('atvi->avti', atvi)

            Htmp[ncore:nocc,:ncore,nocc:,ncore:nocc]  = (- 4 * np.einsum('tuvx,aivx->tiau', self.dm2_cas, self.eri[nocc:,:ncore,ncore:nocc,ncore:nocc]) 
                                                         - 4 * np.einsum('tvux,axvi->tiau', self.dm2_cas, self.eri[nocc:,ncore:nocc,ncore:nocc,:ncore]) 
                                                         - 4 * np.einsum('tvxu,axvi->tiau', self.dm2_cas, self.eri[nocc:,ncore:nocc,ncore:nocc,:ncore]) )
            Htmp[ncore:nocc,:ncore,nocc:,ncore:nocc] += ( 2 * np.einsum('uv,avti->tiau', self.dm1_cas, 4 * avti - aitv - atvi) 
                                                        - 2 * np.einsum('tu,ai->tiau', self.dm1_cas, self.F_core[nocc:,:ncore]) 
                                                        + 2 * np.einsum('tu,ai->tiau', id_cas, F_tot[nocc:,:ncore]) )

            #virtual-active active-core  H_{au,ti}
            Htmp[nocc:,ncore:nocc,ncore:nocc,:ncore]  = np.einsum('auti->tiau', Htmp[ncore:nocc,:ncore,nocc:,ncore:nocc])

        #active-core active-core H_{ti,uj}
        if ncore>0:
            viuj = self.eri[ncore:nocc,:ncore,ncore:nocc,:ncore]
            uivj = self.eri[ncore:nocc,:ncore,ncore:nocc,:ncore]
            uvij = self.eri[ncore:nocc,ncore:nocc,:ncore,:ncore]
            uivj = np.einsum('uivj->viuj', uivj)
            uvij = np.einsum('uvij->viuj', uvij)

            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore]  = 2 * np.einsum('tv,viuj->tiuj', id_cas - self.dm1_cas, 4 * viuj - uivj - uvij)
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] += np.einsum('tiuj->uitj', Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore]) 
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] += ( 4 * np.einsum('utvx,vxij->tiuj', self.dm2_cas, self.eri[ncore:nocc,ncore:nocc,:ncore,:ncore]) 
                                                         + 4 * np.einsum('uxvt,vixj->tiuj', self.dm2_cas, self.eri[ncore:nocc,:ncore,ncore:nocc,:ncore]) 
                                                         + 4  *np.einsum('uxtv,vixj->tiuj', self.dm2_cas, self.eri[ncore:nocc,:ncore,ncore:nocc,:ncore]) )
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] += ( 2 * np.einsum('tu,ij->tiuj', self.dm1_cas, self.F_core[:ncore, :ncore]) 
                                                         - 4 * np.einsum('ij,tvxy,uvxy->tiuj', id_cor, self.dm2_cas, self.eri[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc]) 
                                                         - 2 * np.einsum('ij,uv,tv->tiuj', id_cor, self.dm1_cas, self.F_core[ncore:nocc, ncore:nocc]) )
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] += ( 4 * np.einsum('ij,tu->tiuj', id_cor, F_tot[ncore:nocc, ncore:nocc]) 
                                                         - 4 * np.einsum('tu,ij->tiuj', id_cas, F_tot[:ncore, :ncore]) )

            #AM: I need to think about this
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] = 0.5 * (Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] + np.einsum('tiuj->ujti', Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore]))

        return(Htmp)


    def get_hessianCICI(self):
        ''' This method build the CI-CI part of the hessian '''
        e0 = np.einsum('i,ij,j', self.mat_CI[:,0], self.ham, self.mat_CI[:,0])
        return 2.0 * np.einsum('ki,kl,lj->ij', self.mat_CI[:,1:], self.ham - e0 * np.identity(self.nDet), self.mat_CI[:,1:])


    def get_hessian(self):
        ''' This method concatenate the orb-orb, orb-CI and CI-CI part of the Hessian '''
        H_OrbOrb = (self.get_hessianOrbOrb()[:,:,self.rot_idx])[self.rot_idx,:]
        H_CICI   = self.get_hessianCICI()
        H_OrbCI  = self.get_hessianOrbCI()[self.rot_idx,:]

        return np.block([[H_OrbOrb, H_OrbCI],
                         [H_OrbCI.T, H_CICI]])


    def _eig(self, h, *args):
        return scf.hf.eig(h, None)
    def get_hcore(self, mol=None):
        return self.hcore
    def get_fock(self, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None):
        return mcscf.casci.get_fock(self, mo_coeff, ci, eris, casdm1, verbose)
    def cas_natorb(self, mo_coeff=None, ci=None, eris=None, sort=False, casdm1=None, verbose=None, with_meta_lowdin=True):
        return mcscf.casci.cas_natorb(self, mo_coeff, ci, eris, sort, casdm1, verbose, True)
    def canonicalize_(self):
        # Compute canonicalised natural orbitals
        self.mo_coeff, ci, self.mo_energy = mcscf.casci.canonicalize(self, self.mo_coeff, self.mat_CI[:,0], self.eri, True, True, self.dm1_cas)
        # Insert new "occupied" ci vector
        self.mat_CI[:,0] = ci.ravel()
        # Orthogonalise "unoccupied" ci vectors
        self.mat_CI[:,1:] -= (self.mat_CI[:,[0]].dot(self.mat_CI[:,[0]].T)).dot(self.mat_CI[:,1:])
        self.mat_CI[:,1:] = orthogonalise(self.mat_CI[:,1:], np.identity(self.nDet))
        # Update integrals
        self.update_integrals()
        return

    def get_hessian_index(self):
        hess = self.get_hessian()
        eigs = scipy.linalg.eigvalsh(hess)
        return np.sum(eigs<0)

    def uniq_var_indices(self, nmo, frozen):
        ''' This function creates a matrix of boolean of size (norb,norb). 
            A True element means that this rotation should be taken into 
            account during the optimization. Taken from pySCF.mcscf.casscf '''
        nocc = self.ncore + self.ncas
        mask = np.zeros((self.norb,self.norb),dtype=bool)
        mask[self.ncore:nocc,:self.ncore] = True    # Active-Core rotations
        mask[nocc:,:nocc]                 = True    # Virtual-Core and Virtual-Active rotations
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
                frozen = np.asarray(frozen)
                mask[frozen] = mask[:,frozen] = False
        return mask


##### Main #####
if __name__ == '__main__':

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

    mol = gto.Mole(symmetry=False,unit='B')
    mol.atom = sys.argv[1]
    basis, charge, spin, frozen, cas, grid_option, Hind, maxit = read_config(sys.argv[2])
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()
    myhf = mol.RHF().run()

    # Initialise CAS object
    mycas = ss_casscf(mol, cas[0], cas[1])

    # Set orbital coefficients
    ci_guess = np.identity(4)
    mycas.initialise(myhf.mo_coeff, ci_guess)
    NewtonRaphson(mycas,index=0)
    mycas.canonicalize_()

    print()
    print("  Final energy = {: 16.10f}".format(mycas.energy))
    print("         <S^2> = {: 16.10f}".format(mycas.s2))

    print()
    print("  Canonical natural orbitals:      ")
    print("  ---------------------------------")
    print(" {:^5s}  {:^10s}  {:^10s}".format("  Orb","Occ.","Energy"))
    print("  ---------------------------------")
    for i in range(mycas.ncore + mycas.ncas):
        print(" {:5d}  {: 10.6f}  {: 10.6f}".format(i+1, mycas.mo_occ[i], mycas.mo_energy[i]))
    print("  ---------------------------------")
