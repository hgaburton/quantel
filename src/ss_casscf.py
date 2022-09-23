#!/usr/bin/python3
# Author: Antoine Marie, Hugh G. A. Burton

import numpy as np
import scipy.linalg
from functools import reduce
from pyscf import scf, fci, __config__, ao2mo, lib, mcscf
from pyscf.mcscf import mc_ao2mo
from gnme.cas_noci import cas_proj
from utils import delta_kron, orthogonalise

class ss_casscf():
    def __init__(self, mol, ncas, nelecas, ncore=None):
        self.mol        = mol
        self.nelec      = mol.nelec
        self._scf       = scf.RHF(mol)
        self.verbose    = mol.verbose
        self.stdout     = mol.stdout
        self.max_memory = self._scf.max_memory
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

        # Dimensions of problem 
        self.dim        = self.nrot + self.nDet - 1

        # Define the FCI solver
        self.fcisolver = fci.direct_spin1.FCISolver(mol)

    def copy(self):
        # Return a copy of the current object
        newcas = ss_casscf(self.mol, self.ncas, self.nelecas)
        newcas.initialise(self.mo_coeff.copy(), self.mat_ci.copy())
        return newcas

    def overlap(self, them):
        # Represent the alternative CAS state in the current CI space
        vec2 = cas_proj(self, them, self.ovlp) 
        # Compute the overlap and return
        return np.dot(self.mat_ci[:,0].conj(), vec2)

    def sanity_check(self):
        '''Need to be run at the start of the kernel to verify that the number of 
           orbitals and electrons in the CAS are consistent with the system '''
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
        self.nmo = self.mo_coeff.shape[1]
 
        # Save CI coefficients
        ci_guess = orthogonalise(ci_guess, np.identity(self.nDet)) 
        self.mat_ci = ci_guess

        # Initialise integrals
        self.update_integrals()

    @property
    def energy(self):
        ''' Compute the energy corresponding to a given set of one-el integrals, two-el integrals, 1- and 2-RDM '''
        E  = self.energy_core
        E += np.einsum('pq,pq', self.h1eff, self.dm1_cas)
        E += 0.5 * np.einsum('pqrs,pqrs', self.h2eff, self.dm2_cas)
        return E

    @property
    def s2(self):
        ''' Compute the spin of a given FCI vector '''
        return self.fcisolver.spin_square(self.mat_ci[:,0], self.ncas, self.nelecas)[0]

    @property
    def gradient(self):
        g_orb = self.get_orbital_gradient()
        g_ci  = self.get_ci_gradient()  

        # Unpack matrices/vectors accordingly
        return np.concatenate((g_orb, g_ci))

    @property
    def hessian(self):
        ''' This method concatenate the orb-orb, orb-CI and CI-CI part of the Hessian '''
        self.eri = ao2mo.incore.full(self._scf._eri, self.mo_coeff, compact=False).reshape((self.nmo,)*4)
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



    def update_integrals(self):
        # One-electron Hamiltonian
        self.h1e = np.einsum('ip,ij,jq->pq', self.mo_coeff, self.hcore, self.mo_coeff)

        # Two-electron integrals
        ao2mo_level = getattr(__config__, 'mcscf_mc1step_CASSCF_ao2mo_level', 2)
        self._eri = mc_ao2mo._ERIS(self, self.mo_coeff, method='incore', level=ao2mo_level)
        #self.eri = ao2mo.incore.full(self._scf._eri, self.mo_coeff, compact=False).reshape((self.nmo,)*4)
        #self.eri = np.asarray(self.mol.ao2mo(self.mo_coeff))
        #self.eri = ao2mo.restore(1, self.eri, self.norb)

        # Effective Hamiltonians in CAS space
        self.h1eff, self.energy_core = self.get_h1eff()
        self.h2eff = self.get_h2eff()
        self.h2eff = ao2mo.restore(1, self.h2eff, self.ncas)

        # Reduced density matrices 
        self.dm1_cas, self.dm2_cas = self.get_casrdm_12()

        # Transform 1e integrals
        self.h1e_mo = reduce(np.dot, (self.mo_coeff.T, self.hcore, self.mo_coeff))

        # Fock matrices
        self.F_core, self.F_cas = self.get_fock_matrices()

        # Hamiltonian in active space
        self.ham = self.fcisolver.pspace(self.h1eff, self.h2eff, self.ncas, self.nelecas, np=1000000)[1]

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
        orb_step = np.zeros((self.norb,self.norb))
        orb_step[self.rot_idx] = step
        self.mo_coeff = np.dot(self.mo_coeff, scipy.linalg.expm(orb_step - orb_step.T))

    def rotate_ci(self,step): 
        S       = np.zeros((self.nDet,self.nDet))
        S[1:,0] = step
        self.mat_ci = np.dot(self.mat_ci, scipy.linalg.expm(S - S.T))

    def get_h1eff(self):
        '''CAS sapce one-electron hamiltonian

        Returns:
            A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
            the second is the electronic energy from core.
        '''
        ncas  = self.ncas
        ncore = self.ncore
        nocc  = self.ncore + self.ncas

        # Get core and active orbital coefficients 
        mo_core = self.mo_coeff[:,:ncore]
        mo_cas  = self.mo_coeff[:,ncore:ncore+ncas]

        # Core density matrix (in AO basis)
        self.core_dm = np.dot(mo_core, mo_core.T) * 2

        # Core energy
        energy_core  = self.enuc
        energy_core += np.einsum('ij,ji', self.core_dm, self.hcore)
        energy_core += self._eri.vhf_c[:ncore,:ncore].trace()

        # Get effective Hamiltonian in CAS space
        h1eff  = np.einsum('ki,kl,lj->ij', mo_cas.conj(), self.hcore, mo_cas)
        h1eff += self._eri.vhf_c[ncore:nocc,ncore:nocc]
        return h1eff, energy_core


    def get_h2eff(self):
        '''Compute the active space two-particle Hamiltonian. '''
        nocc  = self.ncore + self.ncas
        ncore = self.ncore
        return self._eri.ppaa[ncore:nocc,ncore:nocc,:,:].copy()


    def get_casrdm_12(self):
        civec = np.reshape(self.mat_ci[:,0],(self.nDeta,self.nDetb))
        dm1_cas, dm2_cas = self.fcisolver.make_rdm12(civec, self.ncas, self.nelecas)
        return dm1_cas, dm2_cas


    def get_spin_dm1(self):
        # Reformat the CI vector
        civec = np.reshape(self.mat_ci[:,0],(self.nDeta,self.nDetb))
        
        # Compute spin density in the active space
        alpha_dm1_cas, beta_dm1_cas = self.fcisolver.make_rdm1s(civec, self.ncas, self.nelecas)
        spin_dens_cas = alpha_dm1_cas - beta_dm1_cas

        # Transform into the AO basis
        mo_cas = self.mo_coeff[:, self.ncore:self.ncore+self.ncas]
        ao_spin_dens  = np.dot(mo_cas, np.dot(spin_dens_cas, mo_cas.T))
        return ao_spin_dens


    def get_fock_matrices(self):
        ''' Compute the core part of the generalized Fock matrix '''
        ncore = self.ncore
        nocc  = self.ncore + self.ncas
       
        # Full Fock
        vj = np.empty((self.nmo,self.nmo))
        vk = np.empty((self.nmo,self.nmo))
        for i in range(self.nmo):
            vj[i] = np.einsum('ij,qij->q', self.dm1_cas, self._eri.ppaa[i])
            vk[i] = np.einsum('ij,iqj->q', self.dm1_cas, self._eri.papa[i])
        fock = self.h1e_mo +  self._eri.vhf_c+vj-vk*0.5

        # Core contribution
        Fcore = self.h1e + self._eri.vhf_c

        # Active space contribution
        Fcas = fock - Fcore

        return Fcore, Fcas


    def get_orbital_gradient(self):
        ''' This method builds the orbital part of the gradient '''
        g_orb = np.zeros((self.norb,self.norb))
        ncore = self.ncore
        ncas  = self.ncas
        nocc  = ncore + ncas
        nvir  = self.norb - nocc
        nmo   = self.nmo

        # Gradient computation from mc1step
        jkcaa = np.empty((nocc,ncas))
        vhf_a = np.empty((nmo,nmo))
        dm2tmp = self.dm2_cas.transpose(1,2,0,3) + self.dm2_cas.transpose(0,2,1,3)
        dm2tmp = dm2tmp.reshape(ncas**2,-1)
        hdm2   = np.empty((nmo,ncas,nmo,ncas))
        g_dm2  = np.empty((nmo,ncas))
        for i in range(nmo):
            jbuf = self._eri.ppaa[i]
            kbuf = self._eri.papa[i]
            if i < nocc: jkcaa[i] = np.einsum('ik,ik->i', 6 * kbuf[:,i] - 2 * jbuf[i], self.dm1_cas)
            vhf_a[i] =(np.einsum('quv,uv->q', jbuf, self.dm1_cas) -
                       np.einsum('uqv,uv->q', kbuf, self.dm1_cas) * 0.5)
            jtmp = lib.dot(jbuf.reshape(nmo,-1), self.dm2_cas.reshape(self.ncas*self.ncas,-1))
            jtmp = jtmp.reshape(nmo,ncas,ncas)
            ktmp = lib.dot(kbuf.transpose(1,0,2).reshape(self.nmo,-1), dm2tmp)
            hdm2[i] = (ktmp.reshape(self.nmo,self.ncas,self.ncas)+jtmp).transpose(1,0,2)
            g_dm2[i] = np.einsum('uuv->v', jtmp[ncore:nocc])
        jbuf = kbuf = jtmp = ktmp = dm2tmp = None
        vhf_ca = self._eri.vhf_c + vhf_a

        g_orb = np.zeros_like(self.h1e_mo)
        g_orb[:,:ncore] = (self.h1e_mo[:,:ncore] + vhf_ca[:,:ncore]) * 2
        g_orb[:,ncore:nocc] = np.dot(self.h1e_mo[:,ncore:nocc]+self._eri.vhf_c[:,ncore:nocc],self.dm1_cas)
        g_orb[:,ncore:nocc] += g_dm2
        return 2.0 * (g_orb - g_orb.T)[self.rot_idx]


    def get_ci_gradient(self):
        ''' This method build the CI part of the gradient '''
        # Return gradient
        return 2.0 * np.einsum('i,ij,jk->k', self.mat_ci[:,0], self.ham, self.mat_ci[:,1:])
    
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

        return t_dm1_cas.T, 0.5*t_dm2_cas

    def get_tCASRDM1(self,ci1,ci2):
        ''' This method compute the 1-electrons transition density matrix between the ci vectors ci1 and ci2 '''
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

        t_dm1_cas = self.fcisolver.trans_rdm1(ci1,ci2,ncas,nelecas)

        return t_dm1_cas.T

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
        two_el = 0
        for i in range(self.nDet):
            for j in range(i,self.nDet):
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
                H_ti[:,:,j,i] = np.conjugate(H_ti[:,:,i,j])

        return H_ai, H_at, H_ti

    def get_approx_ham_commutator(self):
        ''' This method build the Hamiltonian commutator matrices '''
        ncore = self.ncore; ncas = self.ncas; norb = self.norb
        nocc = ncore + ncas; nvir = norb - nocc

        # Initialise output
        H_ai = np.zeros((nvir,ncore,self.nDet,self.nDet))
        H_at = np.zeros((nvir,ncas, self.nDet,self.nDet))
        H_ti = np.zeros((ncas,ncore,self.nDet,self.nDet))

        # Compute contribution for each CI contribution
        mat_id = np.identity(self.nDet)
        two_el = 0
        for i in range(self.nDet):
            for j in range(self.nDet):
                dm1_cas, dm2_cas = self.get_tCASRDM12(mat_id[i],mat_id[j])
                print("CASRDM1_to_RDM1")
                dm1 = self.CASRDM1_to_RDM1(dm1_cas,True)
                print("CASRDM2_to_RDM2")
                dm2 = self.CASRDM2_to_RDM2(dm1_cas,dm2_cas,True)
                if ncore>0 and nvir>0:
                    print("one_el")
                    one_el = ( np.einsum('pa,pi->ai', self.h1e[:,nocc:], dm1[:,:ncore]) 
                             + np.einsum('ap,ip->ai', self.h1e[nocc:,:], dm1[:ncore,:]) )
                    #two_el = ( np.einsum('pars,pirs->ai', self.eri[:,nocc:,:,:], dm2[:,:ncore,:,:]) 
                    #         + np.einsum('aqrs,iqrs->ai', self.eri[nocc:,:,:,:], dm2[:ncore,:,:,:]) 
                    #         + np.einsum('pqra,pqri->ai', self.eri[:,:,:,nocc:], dm2[:,:,:,:ncore]) 
                    #         + np.einsum('pqas,pqis->ai', self.eri[:,:,nocc:,:], dm2[:,:,:ncore,:]) )
                    H_ai[:,:,i,j] = one_el + two_el

                if nvir>0:
                    one_el = ( np.einsum('pa,pt->at', self.h1e[:,nocc:], dm1[:,ncore:nocc]) 
                             + np.einsum('ap,tp->at', self.h1e[nocc:,:], dm1[ncore:nocc,:]) )
                    #two_el = ( np.einsum('pars,ptrs->at', self.eri[:,nocc:,:,:], dm2[:,ncore:nocc,:,:])  
                    #         + np.einsum('aqrs,tqrs->at', self.eri[nocc:,:,:,:], dm2[ncore:nocc,:,:,:]) 
                    #         + np.einsum('pqra,pqrt->at', self.eri[:,:,:,nocc:], dm2[:,:,:,ncore:nocc]) 
                    #         + np.einsum('pqas,pqts->at', self.eri[:,:,nocc:,:], dm2[:,:,ncore:nocc,:]) )
                    H_at[:,:,i,j] = one_el + two_el

                if ncore>0:
                    print("one_el")
                    one_el = ( np.einsum('pt,pi->ti', self.h1e[:,ncore:nocc], dm1[:,:ncore]) 
                             - np.einsum('pi,pt->ti', self.h1e[:,:ncore], dm1[:,ncore:nocc]) 
                             + np.einsum('tp,ip->ti', self.h1e[ncore:nocc,:], dm1[:ncore,:]) 
                             - np.einsum('ip,tp->ti', self.h1e[:ncore,:], dm1[ncore:nocc,:]) )
                    #two_el = ( np.einsum('ptrs,pirs->ti', self.eri[:,ncore:nocc,:,:], dm2[:,:ncore,:,:]) 
                    #         - np.einsum('pirs,ptrs->ti', self.eri[:,:ncore,:,:], dm2[:,ncore:nocc,:,:]) 
                    #         + np.einsum('tqrs,iqrs->ti', self.eri[ncore:nocc,:,:,:], dm2[:ncore,:,:,:]) 
                    #         - np.einsum('iqrs,tqrs->ti', self.eri[:ncore,:,:,:], dm2[ncore:nocc,:,:,:]) 
                    #         + np.einsum('pqrt,pqri->ti', self.eri[:,:,:,ncore:nocc], dm2[:,:,:,:ncore]) 
                    #         - np.einsum('pqri,pqrt->ti', self.eri[:,:,:,:ncore], dm2[:,:,:,ncore:nocc]) 
                    #         + np.einsum('pqts,pqis->ti', self.eri[:,:,ncore:nocc,:], dm2[:,:,:ncore,:]) 
                    #         - np.einsum('pqis,pqts->ti', self.eri[:,:,:ncore,:], dm2[:,:,ncore:nocc,:]) )
                    H_ti[:,:,i,j] = one_el + two_el

        return H_ai, H_at, H_ti


    def get_hessianOrbCI(self):
        ''' This method build the orb-CI part of the hessian '''
        ncore = self.ncore; ncas = self.ncas; nocc = ncore + ncas
        nvir = self.norb - nocc

        H_OCI = np.zeros((self.norb,self.norb,self.nDet-1))
        mat_ci = self.mat_ci

        H_ai, H_at, H_ti = self.get_ham_commutator()

        ci0 = mat_ci[:,0]

        h1e = self.h1e
        eri = self.eri

        for k in range(len(mat_ci)-1): # Loop on Hessian indices
                cleft = mat_ci[:,k+1]
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
            aibj = self.eri[nocc:,:ncore,nocc:,:ncore]
            abij = self.eri[nocc:,nocc:,:ncore,:ncore]

            Htmp[nocc:,:ncore,nocc:,:ncore] = ( 4 * (4 * aibj - abij.transpose((0,2,1,3)) - aibj.transpose((0,3,2,1)))  
                                              + 4 * np.einsum('ij,ab->aibj', id_cor, F_tot[nocc:,nocc:]) 
                                              - 4 * np.einsum('ab,ij->aibj', id_vir, F_tot[:ncore,:ncore]) )

        #virtual-core virtual-active H_{ai,bt}
        if ncore>0 and nvir>0:
            aibv = self.eri[nocc:,:ncore,nocc:,ncore:nocc]
            avbi = self.eri[nocc:,ncore:nocc,nocc:,:ncore]
            abvi = self.eri[nocc:,nocc:,ncore:nocc,:ncore]

            Htmp[nocc:,:ncore,nocc:,ncore:nocc] = ( 2 * np.einsum('tv,aibv->aibt', self.dm1_cas, 4 * aibv - avbi.transpose((0,3,2,1)) - abvi.transpose((0,3,1,2))) 
                                                  - 2 * np.einsum('ab,tvxy,vixy ->aibt', id_vir, 0.5 * self.dm2_cas, self.eri[ncore:nocc, :ncore, ncore:nocc, ncore:nocc]) 
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

            Htmp[nocc:,:ncore,ncore:nocc,:ncore] = ( 2 * np.einsum('tv,aivj->aitj', (2 * id_cas - self.dm1_cas), 4 * aivj - avji.transpose((0,3,1,2)) - ajvi.transpose((0,3,2,1))) 
                                                   - 2 * np.einsum('ji,tvxy,avxy -> aitj', id_cor, 0.5 * self.dm2_cas, self.eri[nocc:,ncore:nocc,ncore:nocc,ncore:nocc]) 
                                                   + 4 * np.einsum('ij,at-> aitj', id_cor, F_tot[nocc:, ncore:nocc]) 
                                                   - 1 * np.einsum('ij,tv,av-> aitj', id_cor, self.dm1_cas, self.F_core[nocc:, ncore:nocc]) )

        #active-core virtual-core H_{tj,ai}
        if ncore>0 and nvir>0:
            Htmp[ncore:nocc, :ncore, nocc:, :ncore] = np.einsum('aitj->tjai',Htmp[nocc:,:ncore,ncore:nocc,:ncore])

        #virtual-active virtual-active H_{at,bu}
        if nvir>0:
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc]  = ( 4 * np.einsum('tuvx,abvx->atbu', 0.5 * self.dm2_cas, self.eri[nocc:,nocc:,ncore:nocc,ncore:nocc]) 
                                                          + 4 * np.einsum('txvu,axbv->atbu', 0.5 * self.dm2_cas, self.eri[nocc:,ncore:nocc,nocc:,ncore:nocc]) 
                                                          + 4 * np.einsum('txuv,axbv->atbu', 0.5 * self.dm2_cas, self.eri[nocc:,ncore:nocc,nocc:,ncore:nocc]) )
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] -= ( 2 * np.einsum('ab,tvxy,uvxy->atbu', id_vir, 0.5 * self.dm2_cas, self.eri[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc]) 
                                                          + 1 * np.einsum('ab,tv,uv->atbu', id_vir, self.dm1_cas, self.F_core[ncore:nocc,ncore:nocc]) )
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] -= ( 2 * np.einsum('ab,uvxy,tvxy->atbu', id_vir, 0.5 * self.dm2_cas, self.eri[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc]) 
                                                          + 1 * np.einsum('ab,uv,tv->atbu', id_vir, self.dm1_cas, self.F_core[ncore:nocc,ncore:nocc]) )
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] +=   2 * np.einsum('tu,ab->atbu', self.dm1_cas, self.F_core[nocc:, nocc:])

        #active-core virtual-active H_{ti,au}
        if ncore>0 and nvir>0:
            avti = self.eri[nocc:, ncore:nocc, ncore:nocc, :ncore]
            aitv = self.eri[nocc:, :ncore, ncore:nocc, ncore:nocc]

            Htmp[ncore:nocc,:ncore,nocc:,ncore:nocc]  = (- 4 * np.einsum('tuvx,aivx->tiau', 0.5 * self.dm2_cas, self.eri[nocc:,:ncore,ncore:nocc,ncore:nocc]) 
                                                         - 4 * np.einsum('tvux,axvi->tiau', 0.5 * self.dm2_cas, self.eri[nocc:,ncore:nocc,ncore:nocc,:ncore]) 
                                                         - 4 * np.einsum('tvxu,axvi->tiau', 0.5 * self.dm2_cas, self.eri[nocc:,ncore:nocc,ncore:nocc,:ncore]) )
            Htmp[ncore:nocc,:ncore,nocc:,ncore:nocc] += ( 2 * np.einsum('uv,avti->tiau', self.dm1_cas, 4 * avti - aitv.transpose((0,3,2,1)) - avti.transpose((0,2,1,3)) ) 
                                                        - 2 * np.einsum('tu,ai->tiau', self.dm1_cas, self.F_core[nocc:,:ncore]) 
                                                        + 2 * np.einsum('tu,ai->tiau', id_cas, F_tot[nocc:,:ncore]) )

            #virtual-active active-core  H_{au,ti}
            Htmp[nocc:,ncore:nocc,ncore:nocc,:ncore]  = np.einsum('auti->tiau', Htmp[ncore:nocc,:ncore,nocc:,ncore:nocc])

        #active-core active-core H_{ti,uj}
        if ncore>0:
            viuj = self.eri[ncore:nocc,:ncore,ncore:nocc,:ncore]
            uvij = self.eri[ncore:nocc,ncore:nocc,:ncore,:ncore]

            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore]  = 2 * np.einsum('tv,viuj->tiuj', id_cas - self.dm1_cas, 4 * viuj - viuj.transpose((2,1,0,3)) - uvij.transpose((1,2,0,3)) )
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] += np.einsum('tiuj->uitj', Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore]) 
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] += ( 4 * np.einsum('utvx,vxij->tiuj', 0.5 * self.dm2_cas, self.eri[ncore:nocc,ncore:nocc,:ncore,:ncore]) 
                                                         + 4 * np.einsum('uxvt,vixj->tiuj', 0.5 * self.dm2_cas, self.eri[ncore:nocc,:ncore,ncore:nocc,:ncore]) 
                                                         + 4  *np.einsum('uxtv,vixj->tiuj', 0.5 * self.dm2_cas, self.eri[ncore:nocc,:ncore,ncore:nocc,:ncore]) )
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] += ( 2 * np.einsum('tu,ij->tiuj', self.dm1_cas, self.F_core[:ncore, :ncore]) 
                                                         - 4 * np.einsum('ij,tvxy,uvxy->tiuj', id_cor, 0.5 * self.dm2_cas, self.eri[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc]) 
                                                         - 2 * np.einsum('ij,uv,tv->tiuj', id_cor, self.dm1_cas, self.F_core[ncore:nocc, ncore:nocc]) )
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] += ( 4 * np.einsum('ij,tu->tiuj', id_cor, F_tot[ncore:nocc, ncore:nocc]) 
                                                         - 4 * np.einsum('tu,ij->tiuj', id_cas, F_tot[:ncore, :ncore]) )

            #AM: I need to think about this
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] = 0.5 * (Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] + np.einsum('tiuj->ujti', Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore]))

        return(Htmp)


    def get_hessianCICI(self):
        ''' This method build the CI-CI part of the hessian '''
        e0 = np.einsum('i,ij,j', self.mat_ci[:,0], self.ham, self.mat_ci[:,0])
        return 2.0 * np.einsum('ki,kl,lj->ij', self.mat_ci[:,1:], self.ham - e0 * np.identity(self.nDet), self.mat_ci[:,1:])

    def get_metric(self):
        met_CICI   = self.get_metricCICI()
        print(self.rot_idx)
        met_OrbCI  = self.get_metricOrbCI()[self.rot_idx,:]
        met_OrbOrb = self.get_metricOrbOrb()[self.rot_idx,:][:,self.rot_idx]

        return np.block([[met_OrbOrb,  met_OrbCI],
                         [met_OrbCI.T, met_CICI]])

    def get_metricOrbOrb(self):

        norb = self.norb; ncore = self.ncore; ncas = self.ncas
        nocc = ncore + ncas; nvir = norb - nocc

        met = np.zeros((norb,norb,norb,norb))
        dm1_cas, dm2_cas = self.get_casrdm_12()

        if ncore>0 and nvir>0:
            for i in range(ncore):
                for a in range(nocc,norb):
                    met[a,i,a,i] = 1

        if nvir>0:
            for a in range(nocc,norb):
                for t in range(ncore,nocc):
                    for u in range(ncore,nocc):
                        dm_tt = dm1_cas[t-ncore,t-ncore]
                        dm_uu = dm1_cas[u-ncore,u-ncore]
                        met[a,t,a,u] = dm1_cas[t-ncore,u-ncore]/np.sqrt(dm_tt * dm_uu)

        if ncore>0:
            for i in range(ncore):
                for t in range(ncore,nocc):
                    for u in range(ncore,nocc):
                        m_t = 2 - dm1_cas[t-ncore,t-ncore]
                        m_u = 2 - dm1_cas[u-ncore,u-ncore]
                        met[i,t,i,u] = (2*delta_kron(t,u) - dm1_cas[u-ncore,t-ncore])/np.sqrt(m_t * m_u)

        return met
        

    def get_metricOrbCI(self):
        '''Build the orbital-CI component of the metric'''
        met = np.zeros((self.norb,self.norb,self.nDet-1))

        mat_ci = self.mat_ci
        ci0 = mat_ci[:,0]

        for k in range(len(mat_ci)-1):
            ciK = mat_ci[:,k+1]
            dm1_cas = self.get_tCASRDM1(ciK, ci0)
            dm1 = self.CASRDM1_to_RDM1(dm1_cas,True)
            met[:,:,k] = dm1 - dm1.T
            print(met[:,:,k])

        return met

    def get_metricCICI(self):
        return np.identity(self.nDet-1)

    def _eig(self, h, *args):
        return scf.hf.eig(h, None)
    def get_hcore(self, mol=None):
        return self.hcore
    def get_fock(self, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None):
        return mcscf.casci.get_fock(self, mo_coeff, ci, eris, casdm1, verbose)
    def cas_natorb(self, mo_coeff=None, ci=None, eris=None, sort=False, casdm1=None, verbose=None, with_meta_lowdin=True):
        test = mcscf.casci.cas_natorb(self, mo_coeff, ci, eris, sort, casdm1, verbose, True)
        return test
    def canonicalize_(self):
        # Compute canonicalised natural orbitals
        self.mo_coeff, ci, self.mo_energy = mcscf.casci.canonicalize(self, self.mo_coeff, ci=self.mat_ci[:,0], 
                                                       eris=self._eri, sort=True, cas_natorb=True, casdm1=self.dm1_cas)

        # Insert new "occupied" ci vector
        self.mat_ci[:,0] = ci.ravel()
        self.mat_ci = orthogonalise(self.mat_ci, np.identity(self.nDet))

        # Update integrals
        self.update_integrals()
        return

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
    import sys, re, os
    from newton_raphson import NewtonRaphson
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
