#!/usr/bin/env python
# Author: Antoine Marie

import sys

from functools import reduce
import numpy as np
import scipy.linalg
import pyscf
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf import mcscf
from pyscf.mcscf import casci
from pyscf import fci
from pyscf.fci import spin_op


##### Definition of the class #####

class NR_CASSCF(lib.StreamObject):
    '''NR-CASSCF

    Args:
        myhf_or_mol : SCF object or Mole object
            SCF or Mole to define the problem size.
        ncas : int
            Number of active orbitals.
        nelecas : int or a pair of int
            Number of electrons in active space.

    Kwargs:
        ncore : int
            Number of doubly occupied core orbitals. If not presented, this
            parameter can be automatically determined.

    Attributes:

    Saved results:

    ''' #TODO Write the documentation

    def __init__(self,myhf_or_mol,ncas,nelecas,ncore=None,initMO=None,initCI=None,frozen=None): #TODO Initialize the argument and the attributes
        ''' The init method is ran when an instance of the class is created to initialize all the args, kwargs and attributes
        '''
        if isinstance(myhf_or_mol, gto.Mole):   # Check if the arg is an HF object or a molecule object
            myhf = scf.RHF(myhf_or_mol)
        else:
            myhf = myhf_or_mol

            mol = myhf.mol
        self.mol = mol                          # Molecule object
        self.nelec = mol.nelec                  # Number of electrons
        self._scf = myhf                        # SCF object
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = myhf.max_memory
        self.ncas = ncas                        # Number of active orbitals
        if isinstance(nelecas, (int, np.integer)):
            nelecb = (nelecas-mol.spin)//2
            neleca = nelecas - nelecb
            self.nelecas = (neleca, nelecb)     # Tuple of number of active electrons
        else:
            self.nelecas = (nelecas[0],nelecas[1]).astype(int)
        self.v1e = mol.intor('int1e_nuc')       # Nuclear repulsion matrix elements
        self.t1e = mol.intor('int1e_kin')       # Kinetic energy matrix elements
        self.h1e_AO =  self.t1e + self.v1e      # 1-electron matrix elements in the AO basis
        self.norb = len(self.h1e_AO)            # Number of orbitals
        self.nDeta = (scipy.special.comb(self.ncas,self.nelecas[0])).astype(int)
        self.nDetb = (scipy.special.comb(self.ncas,self.nelecas[1])).astype(int)
        self.nDet = (self.nDeta*self.nDetb).astype(int)
        self.eri_AO = mol.intor('int2e')        # ERI in the AO basis in the chemist notation
        self._ncore = ncore                     # Number of core orbitals
        self._initMO = initMO                   # Initial MO coefficients
        self._initCI = initCI                   # Initial CI coefficients
        self.frozen = frozen                    # Number of frozen orbitals
        self.mo_coeff = None                    # MO coeff at this stage of the calculation
        self.mat_CI = None                      # CI coeff at this stage of the calculation
        self.h1e = None                         # 1-electron matrix elements in the MO basis
        self.eri = None                         # ERI in the MO basis in the chemist notation

        self.fcisolver = fci.direct_spin1.FCISolver(mol)

    @property
    def ncore(self):
        ''' Initialize the number of core orbitals '''
        if self._ncore is None:
            ncorelec = self.mol.nelectron - sum(self.nelecas)
            assert ncorelec % 2 == 0
            assert ncorelec >= 0
            return ncorelec // 2
        else:
            return self._ncore

    @property
    def initMO(self):
        ''' Initialize the MO coefficients and MO 1- and 2-electrons integrals matrix elements '''
        if self._initMO is None:
            self.mo_coeff = self._scf.mo_coeff
            self.h1e = np.einsum('ip,ij,jq->pq', self._scf.mo_coeff, self.h1e_AO, self._scf.mo_coeff) # We transform the 1-electron integrals to the MO basis
            self.eri = np.asarray(mol.ao2mo(self._scf.mo_coeff)) # eri in the MO basis as super index matrix (ij|kl) with i>j and k>l VERIFY THIS LAST POINT
            self.eri = ao2mo.restore(1, self.eri, self.norb) # eri in the MO basis with chemist notation
            return self._scf.mo_coeff
        else:
            self.mo_coeff = initMO
            self.h1e = np.einsum('ip,ij,jq->pq', self.mo_coeff, self.h1e_AO, self.mo_coeff) # We transform the 1-electron integrals to the MO basis
            self.eri = np.asarray(mol.ao2mo(self.mo_coeff)) # eri in the MO basis as super index matrix (ij|kl) with i>j and k>l VERIFY THIS LAST POINT
            self.eri = ao2mo.restore(1, self.eri, norb) # eri in the MO basis with chemist notation
            return self._initMO

    @property #TODO add an option to initialize as a full CASCI diagonalization instead of a diagonal of 1
    def initCI(self):
        ''' Initialize the CI coefficients '''
        if self._initCI is None:
            self.mat_CI = np.identity(self.nDet, dtype="float")
            return np.identity(self.nDet, dtype="float")
        else:
            self.mat_CI = self._initCI
            return self._initCI

    def dump_flags(self): #TODO
        '''
        '''
        pass

    def check_sanity(self):
        assert self.ncas > 0
        ncore = self.ncore
        nvir = self.mo_coeff.shape[1] - ncore - self.ncas
        assert ncore >= 0
        assert nvir >= 0
        assert ncore * 2 + sum(self.nelecas) == self.mol.nelectron
        assert 0 <= self.nelecas[0] <= self.ncas
        assert 0 <= self.nelecas[1] <= self.ncas
        return self

    def energy_nuc(self):
        ''' Retrieve the nuclear energy from the SCF instance '''
        return self._scf.energy_nuc()

    def get_hcore(self, mol=None):
        ''' Retrieve the 1 electron Hamiltonian from the SCF instance '''
        return self._scf.get_hcore(mol)

    def get_CASRDM_1(self,ci):
        ''' This method compute the 1-RDM in the CAS space of a given ci wave function '''
        ncas = self.ncas
        nelecas = self.nelecas
        if len(ci.shape)==1:
            nDeta = scipy.special.comb(ncas,nelecas[0]).astype(int)
            nDetb = scipy.special.comb(ncas,nelecas[1]).astype(int)
            ci = ci.reshape((nDeta,nDetb)) #TODO be careful that the reshape is the inverse of the "unfolding" of the matrix vector
        dm1_cas = self.fcisolver.make_rdm1(ci,ncas,nelecas) # the make_rdm1 method takes a ci vector as a matrix na x nb
        return dm1_cas.T # We transpose the 1-RDM because their convention is <|a_q^\dagger a_p|>


    def get_CASRDM_12(self,ci):
        ''' This method compute the 1-RDM and 2-RDM in the CAS space of a given ci wave function '''
        ncas = self.ncas
        nelecas = self.nelecas
        if len(ci.shape)==1:
            nDeta = scipy.special.comb(ncas,nelecas[0]).astype(int)
            nDetb = scipy.special.comb(ncas,nelecas[1]).astype(int)
            ci = ci.reshape((nDeta,nDetb)) #TODO be careful that the reshape is the inverse of the "unfolding" of the matrix vector

        dm1_cas, dm2_cas = self.fcisolver.make_rdm12(ci,ncas,nelecas) # The make_rdm12 method takes a ci vector as a matrix na x nb

        return dm1_cas.T, dm2_cas # We transpose the 1-RDM because their convention is <|a_q^\dagger a_p|>

    def CASRDM1_to_RDM1(self,dm1_cas):
        ''' This method takes a 1-RDM in the CAS space and transform it to the full MO space '''
        ncore = self.ncore
        ncas = self.ncas
        dm1 = np.zeros((self.norb,self.norb))
        if ncore > 0:
            dm1_core = 2*np.identity(self.ncore, dtype="int") #the OccOcc part of dm1 is a diagonal of 2
            dm1[:ncore,:ncore] = dm1_core
        dm1[ncore:ncore+ncas,ncore:ncore+ncas] = dm1_cas
        return dm1

    @staticmethod
    def dm2_mo_occ(part,p,q,r,s,dm1):
        ''' This function compute the core part of the 2-RDM. The ijkl, ijpq and pijq can be simplified according to the following equations (Eq(62)-(67)). The piqj is zero as well as all the elements with a virtual index or an odd number of core indices. '''
        if part=='ijkl':
            return(2*delta_kron(p,q)*np.kron(r,s) - delta_kron(p,r)*np.kron(q,s))
        elif part=='ijpq':
            return(delta_kron(p,q)*dm1[r,s] - delta_kron(p,s)*delta_kron(q,r))
        elif part=='pijq':
            return(2*delta_kron(q,p)*delta_kron(r,s) - 0.5*delta_kron(q,r)*dm1[p,q])
        else:
            return('Wrong specification of part')

    def CASRDM2_to_RDM2(self,dm1_cas,dm2_cas):
        ''' This method takes a 2-RDM in the CAS space and transform it to the full MO space '''
        ncore = self.ncore
        ncas = self.ncas
        norb = self.norb
        nocc = ncore + ncas
        dm2 = np.zeros((norb,norb,norb,norb))
        for i in range(ncore):      # really not elegant ... find an alternative way to do this
            for j in range(ncore):
                for k in range(ncore):
                    for l in range(ncore):
                        dm2[i,j,k,l] = self.dm2_mo_occ('ijkl',i,j,k,l,dm1_cas)
        for i in range(ncore):
            for j in range(ncore):
                for p in range(ncas):
                    for q in range(ncas):
                        dm2[i,j,p+ncore,q+ncore] = self.dm2_mo_occ('ijpq',i,j,p,q,dm1_cas)
        for i in range(ncore):
            for j in range(ncore):
                for p in range(ncas):
                    for q in range(ncas):
                        dm2[p+ncore,i,j,q+ncore] = self.dm2_mo_occ('pijq',p,i,j,q,dm1_cas)

        dm2[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc] = dm2_cas # Finally we add the uvxy sector
        return dm2

    def get_tCASRDM1(self,ci1,ci2):
        ''' This method compute the 1-electron transition density matrix between the ci vectors ci1 and ci2 '''
        ncas = self.ncas
        nelecas = self.nelecas
        if len(ci1.shape)==1:
            nDeta = scipy.special.comb(ncas,nelecas[0]).astype(int)
            nDetb = scipy.special.comb(ncas,nelecas[1]).astype(int)
            ci1 = ci1.reshape((nDeta,nDetb)) #TODO be careful that the reshape is the inverse of the "unfolding" of the matrix vector
        if len(ci2.shape)==1:
            nDeta = scipy.special.comb(ncas,nelecas[0]).astype(int)
            nDetb = scipy.special.comb(ncas,nelecas[1]).astype(int)
            ci2 = ci2.reshape((nDeta,nDetb)) #TODO be careful that the reshape is the inverse of the "unfolding" of the matrix vector

        t_dm1_cas = mycas.fcisolver.trans_rdm1(ci1,ci2,ncas,nelecas)

        return t_dm1_cas #TODO do we need to transpose this matrix

    def get_tCASRDM12(self,ci1,ci2):
        ''' This method compute the 1- and 2-electrons transition density matrix between the ci vectors ci1 and ci2 '''
        ncas = self.ncas
        nelecas = self.nelecas
        if len(ci1.shape)==1:
            nDeta = scipy.special.comb(ncas,nelecas[0]).astype(int)
            nDetb = scipy.special.comb(ncas,nelecas[1]).astype(int)
            ci1 = ci1.reshape((nDeta,nDetb)) #TODO be careful that the reshape is the inverse of the "unfolding" of the matrix vector
        if len(ci2.shape)==1:
            nDeta = scipy.special.comb(ncas,nelecas[0]).astype(int)
            nDetb = scipy.special.comb(ncas,nelecas[1]).astype(int)
            ci2 = ci2.reshape((nDeta,nDetb)) #TODO be careful that the reshape is the inverse of the "unfolding" of the matrix vector
        t_dm1_cas, t_dm2_cas = mycas.fcisolver.trans_rdm12(ci1,ci2,ncas,nelecas)

        return t_dm1_cas, t_dm2_cas #TODO do we need to transpose this matrix

    def get_F_core(self):
        ncore = self.ncore
        return(self.h1e + 2*np.einsum('pqii->pq', self.eri[:, :, :ncore, :ncore]) - np.einsum('piiq->pq', self.eri[:, :ncore, :ncore, :]))

    def get_F_cas(self,dm1_cas):
        ncore = self.ncore
        nocc = ncore + self.ncas
        return(np.einsum('tu,pqtu->pq', dm1_cas, self.eri[:, :, ncore:nocc, ncore:nocc]) - 0.5*np.einsum('tu,putq->pq', dm1_cas, self.eri[:, ncore:nocc, ncore:nocc, :]))

    def get_genFock(self,dm1_cas):
        ''' This method build the generalized Fock matrix '''
        return self.get_F_core() + self.get_F_cas(dm1_cas)

    # def get_genFockAO(self, dm1_cas):
    #     ncore = self.ncore
    #     nocc = ncore + self.ncas
    #     mo_coeff = self.mo_coeff
    #     dm1_cas_AO = np.einsum('pq,qr,sr',mo_coeff, dm1_cas, mo_coeff)
    #
    #     F_core = self.h1e_AO + 2*np.einsum('pqii->pq', self.eri_AO[:, :, :ncore, :ncore]) - np.einsum('piiq->pq', self.eri_AO[:, :ncore, :ncore, :])
    #     F_cas = np.einsum('tu,pqtu->pq', dm1_cas_AO, self.eri_AO[:, :, ncore:nocc, ncore:nocc]) - 0.5*np.einsum('tu,putq->pq', dm1_cas_AO, self.eri_AO[:, ncore:nocc, ncore:nocc, :])
    #
    #     return

    def get_gradOrb(self,dm1_cas,dm2_cas):
        ''' This method build the orbital part of the gradient '''
        g_orb = np.zeros((self.norb,self.norb))
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        nvir = self.norb - nocc
        F_core = self.get_F_core()
        F_cas = self.get_F_cas(dm1_cas)

        #virtual-core rotations g_{ai}
        if ncore>0 and nvir>0:
            g_orb[nocc:,:ncore] = 4*(F_core + F_cas)[nocc:,:ncore]
        #active-core rotations g_{ti}
        if ncore>0:
            g_orb[ncore:nocc,:ncore] = 4*(F_core + F_cas)[ncore:nocc,:ncore] - 2*np.einsum('tv,iv->ti', dm1_cas, F_core[:ncore,ncore:nocc]) - 2*np.einsum('tvxy,ivxy->ti', dm2_cas, self.eri[:ncore,ncore:nocc,ncore:nocc,ncore:nocc])
        #virtual-active rotations g_{at}
        if nvir>0:
            g_orb[nocc:,ncore:nocc] = 2*np.einsum('tv,av->at', dm1_cas, F_core[nocc:,ncore:nocc]) + 2*np.einsum('tvxy,avxy->at',dm2_cas, self.eri[nocc:,ncore:nocc,ncore:nocc,ncore:nocc])

        return g_orb - g_orb.T # this gradient is a matrix, be careful we need to pack it before joining it with the CI part

    def get_gradCI(self):
        ''' This method build the CI part of the gradient '''
        mat_CI = self.mat_CI
        g_CI = np.zeros(len(mat_CI)-1)
        ciO = mat_CI[:,0]
        h1e = self.h1e
        eri = self.eri

        for i in range(len(mat_CI)-1):
            t_dm1_cas, t_dm2_cas = self.get_tCASRDM12(mat_CI[:,i+1],ciO)
            t_dm1 = self.CASRDM1_to_RDM1(t_dm1_cas)
            t_dm2 = self.CASRDM2_to_RDM2(t_dm1_cas,t_dm2_cas)

            g_CI[i] = 2*np.einsum('pq,pq',h1e,t_dm1) + np.einsum('pqrs,pqrs',eri,t_dm2)

        return g_CI

    def form_grad(self,g_orb,g_ci):
        ''' This method concatenate the orbital and CI part of the gradient '''
        uniq_g_orb = self.pack_uniq_var(g_orb) #We apply the mask to obtain a list of unique rotations
        g = np.concatenate((uniq_g_orb,g_ci))
        return g

    def uniq_var_indices(self, nmo, frozen):
        ''' This function creates a matrix of boolean of size (norb,norb). A True element means that this rotation should be taken into account during the optimization. Taken from pySCF.mcscf.casscf '''
        norb = self.norb
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        mask = np.zeros((norb,norb),dtype=bool)
        mask[ncore:nocc,:ncore] = True # Active-Core rotations
        mask[nocc:,:nocc] = True # Virtual-Core and Virtual-Active rotations
        # if self.internal_rotation:
        #    mask[ncore:nocc,ncore:nocc][np.tril_indices(ncas,-1)] = True
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
                frozen = np.asarray(frozen)
                mask[frozen] = mask[:,frozen] = False
        return mask

    def pack_uniq_var(self, mat):
        ''' This method transforms a matrix of rotations K into a list of unique rotations elements. Taken from pySCF.mcscf.casscf '''
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo,self.frozen)
        return mat[idx]

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        ''' This method transforms a list of unique rotations elements into an anti-symmetric rotation matrix. Taken from pySCF.mcscf.casscf '''
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        mat = np.zeros((nmo,nmo))
        mat[idx] = v
        return mat - mat.T

    def get_hessianOrbOrb(self,dm1_cas,dm2_cas):
        ''' This method build the orb-orb part of the hessian '''
        norb = self.norb
        H = np.zeros((norb,norb,norb,norb))
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        nvir = norb - nocc
        eri = self.eri
        F_core = self.get_F_core()
        F_cas = self.get_F_cas(dm1_cas)
        F_tot = F_core + F_cas

        #virtual-core virtual-core H_{ai,bj}
        if ncore>0 and nvir>0:
            tmp = eri[nocc:, :ncore, nocc:, :ncore]

            aibj = eri[nocc:, :ncore, nocc:, :ncore]
            abij = eri[nocc:, nocc:, :ncore, :ncore]
            ajbi = eri[nocc:, :ncore, nocc:, :ncore]

            abij = np.einsum('abij->aibj', abij)
            ajbi = np.einsum('ajbi->aibj', ajbi)

            H[nocc:, :ncore, nocc:, :ncore] = 4*(4*aibj - abij - ajbi) + 4*np.einsum('ij,ab->aibj',np.identity(ncore),F_tot[nocc:,nocc:]) - 4*np.einsum('ab,ij->aibj', np.identity(nvir), F_tot[:ncore,:ncore])

        #virtual-core virtual-active H_{ai,bt}
        if ncore>0 and nvir>0:
            aibv = eri[nocc:, :ncore, nocc:, ncore:nocc]
            avbi = eri[nocc:, ncore:nocc, nocc:, :ncore]
            abvi = eri[nocc:, nocc:, ncore:nocc, :ncore]

            avbi = np.einsum('avbi->aibv', avbi)
            abvi = np.einsum('abvi->aibv', abvi)

            H[nocc:, :ncore, nocc:, ncore:nocc] = 2*np.einsum('tv,aibv->aibt', dm1_cas, 4*aibv - avbi - abvi) - np.einsum('ab,tvxy,vixy ->aibt', np.identity(nvir), dm2_cas,eri[ncore:nocc, :ncore, ncore:nocc, ncore:nocc]) - 2*np.einsum('ab,ti->aibt', np.identity(nvir), F_tot[ncore:nocc, :ncore]) - np.einsum('ab,tv,vi->aibt', np.identity(nvir), dm1_cas, F_core[ncore:nocc, :ncore])

            H[nocc:, ncore:nocc, nocc:, :ncore] =  np.einsum('aibt->btai', H[nocc:, :ncore, nocc:, ncore:nocc])

        #virtual-core active-core H_{ai,tj}
        if ncore>0 and nvir>0:
            aivj = eri[nocc:, :ncore, ncore:nocc, :ncore]
            avji = eri[nocc:, ncore:nocc, :ncore, :ncore]
            ajvi = eri[nocc:, :ncore, ncore:nocc, :ncore]

            avji = np.einsum('avji->aivj', avji)
            ajvi = np.einsum('ajvi->aivj', ajvi)

            H[nocc:, :ncore, ncore:nocc, :ncore] = 2*np.einsum('tv,aivj->aitj', (2*np.identity(ncas) - dm1_cas), 4*aivj - avji - ajvi) - np.einsum('ji,tvxy,avxy -> aitj', np.identity(ncore), dm2_cas, eri[nocc:, ncore:nocc, ncore:nocc, ncore:nocc]) + 4*np.einsum('ij,at-> aitj', np.identity(ncore), F_tot[nocc:, ncore:nocc]) - np.einsum('ij,tv,av-> aitj', np.identity(ncore), dm1_cas, F_core[nocc:, ncore:nocc])

            H[ncore:nocc, :ncore, nocc:, :ncore] =  np.einsum('aitj->tjai', H[nocc:, :ncore, ncore:nocc, :ncore])

        #active-core active-core H_{ti,uj}
        if ncore>0:
            tmp1 = 2*np.einsum('utvx,vxij->tiuj', dm2_cas, eri[ncore:nocc, ncore:nocc, :ncore, :ncore]) + 2*np.einsum('uxvt,vixj->tiuj', dm2_cas, eri[ncore:nocc, :ncore, ncore:nocc, :ncore]) + 2*np.einsum('uxtv,vixj->tiuj', dm2_cas, eri[ncore:nocc, :ncore, ncore:nocc, :ncore])

            viuj = eri[ncore:nocc, :ncore, ncore:nocc, :ncore]
            uivj = eri[ncore:nocc, :ncore, ncore:nocc, :ncore]
            uvij = eri[ncore:nocc, ncore:nocc, :ncore, :ncore]
            uivj = np.einsum('uivj->viuj', uivj)
            uvij = np.einsum('uvij->viuj', uvij)
            tmp2 = 2*np.einsum('tv,viuj->tiuj', np.identity(ncas) - dm1_cas, 4*viuj - uivj - uvij)
            tmp2 = tmp2 + np.einsum('tiuj->uitj', tmp2)

            tmp3 = 2*np.einsum('tu,ij->tiuj', dm1_cas, F_core[:ncore,:ncore]) - 2*np.einsum('ij,tvxy,uvxy->tiuj', np.identity(ncore), dm2_cas, eri[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc]) - 2*np.einsum('ij,uv,tv->tiuj', np.identity(ncore), dm1_cas, F_core[ncore:nocc, ncore:nocc]) + 4*np.einsum('ij,tu->tiuj', np.identity(ncore), F_tot[ncore:nocc, ncore:nocc]) - 4*np.einsum('tu,ij->tiuj', np.identity(ncas), F_tot[:ncore,:ncore])

            H[ncore:nocc, :ncore, ncore:nocc, :ncore] = tmp1 + tmp2 + tmp3

        #active-core virtual-active H_{ti,au}
        if ncore>0 and nvir>0:
            dm2_cas_tvux = np.einsum('tuvx->tvux',dm2_cas)
            dm2_cas_tvxu = np.einsum('tuvx->tvxu',dm2_cas)
            tmp1 = - 2*np.einsum('tuvx,aivx->tiau', dm2_cas, eri[nocc:, :ncore, ncore:nocc, ncore:nocc]) - 2*np.einsum('tvux,axvi->tiau', dm2_cas, eri[nocc:, ncore:nocc, ncore:nocc, :ncore]) - 2*np.einsum('tvxu,axvi->tiau', dm2_cas, eri[nocc:, ncore:nocc, ncore:nocc, :ncore])

            avti = eri[nocc:, ncore:nocc, ncore:nocc, :ncore]
            aitv = eri[nocc:, :ncore, ncore:nocc, ncore:nocc]
            atvi = eri[nocc:, ncore:nocc, ncore:nocc, :ncore]

            aitv = np.einsum('aitv->avti', aitv)
            atvi = np.einsum('atvi->avti', atvi)

            tmp2 = 2*np.einsum('uv,avti->tiau', dm1_cas, 4*avti - aitv - atvi) - 2*np.einsum('tu,ai->tiau', dm1_cas, F_core[nocc:, :ncore]) + 2*np.einsum('tu,ai->tiau',np.identity(ncas),F_tot[nocc:,:ncore])

            H[ncore:nocc, :ncore, nocc:, ncore:nocc] = tmp1 + tmp2

            H[nocc:, ncore:nocc, ncore:nocc, :ncore] =  np.einsum('tiau->auti', H[ncore:nocc, :ncore, nocc:, ncore:nocc])

        #virtual-active virtual-active H_{at,bu}
        if nvir>0:
            dm2_cas_txvu = np.einsum('tuvx->txvu',dm2_cas)
            dm2_cas_txuv = np.einsum('tuvx->txuv',dm2_cas)
            tmp1 = 2*np.einsum('tuvx,abvx->atbu', dm2_cas, eri[nocc:, nocc:, ncore:nocc, ncore:nocc]) + 2*np.einsum('txvu,axbv->atbu', dm2_cas_txvu, eri[nocc:, ncore:nocc, nocc:, ncore:nocc]) + 2*np.einsum('txuv,axbv->atbu', dm2_cas_txuv, eri[nocc:, ncore:nocc, nocc:, ncore:nocc])


            tmp2 = - np.einsum('ab,tvxy,uvxy->atbu',np.identity(nvir),dm2_cas,eri[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc]) - np.einsum('ab,tv,uv->atbu',np.identity(nvir),dm1_cas,F_core[ncore:nocc,nocc:])

            H[nocc:, ncore:nocc, nocc:, ncore:nocc] = tmp1 + tmp2 + 2*np.einsum('atbu->aubt',tmp2) + 2*np.einsum('tu,ab->atbu', dm1_cas, F_core[nocc:, nocc:])


        # symmetrize the Hessian
        H = H + np.einsum('pqrs->qpsr', H)

        return(H)

    def get_hamiltonian(self):
        ''' This method build the Hamiltonian matrix '''
        H = np.zeros((self.nDet,self.nDet))
        id = np.identity(self.nDet)
        for i in range(self.nDet):
            for j in range(self.nDet):
                dm1_cas, dm2_cas = self.get_tCASRDM12(id[i],id[j])
                dm1 = self.CASRDM1_to_RDM1(dm1_cas)
                dm2 = self.CASRDM2_to_RDM2(dm1_cas,dm2_cas)
                H[i,j] = np.einsum('pq,pq',self.h1e,dm1) + 0.5*np.einsum('pqrs,pqrs',self.eri,dm2)
        return H

    def get_hessianCICI(self):
        ''' This method build the CI-CI part of the hessian '''
        mat_CI = self.mat_CI
        hessian_CICI = np.zeros((len(mat_CI)-1,len(mat_CI)-1))
        H = self.get_hamiltonian()
        ciO = mat_CI[:,0]
        eO = np.einsum('i,ij,j',ciO,H,ciO)
        h1e = self.h1e
        eri = self.eri

        for k in range(1,len(mat_CI)): # Loop on Hessian indices
                cleft = mat_CI[:,k]
                for l in range(1,len(mat_CI)):
                    cright = mat_CI[:,l]
                    hessian_CICI[k-1,l-1] = 2*np.einsum('i,ij,j',cleft, H, cright) - 2*delta_kron(k,l)*eO

        return hessian_CICI

    def get_hamiltonianComm(self):
        ''' This method build the Hamiltonian commutator matrices '''
        ncore = self.ncore
        ncas = self.ncas
        norb = self.norb
        nocc = ncore + ncas
        nvir = norb - nocc

        H_ai = np.zeros((nvir,ncore,self.nDet,self.nDet))
        H_at = np.zeros((nvir,ncas,self.nDet,self.nDet))
        H_ti = np.zeros((ncas,ncore,self.nDet,self.nDet))

        h1e = self.h1e
        eri = self.eri

        id = np.identity(self.nDet)
        for i in range(self.nDet):
            for j in range(self.nDet):
                dm1_cas, dm2_cas = self.get_tCASRDM12(id[i],id[j])
                dm1 = self.CASRDM1_to_RDM1(dm1_cas)
                dm2 = self.CASRDM2_to_RDM2(dm1_cas,dm2_cas)
                if ncore>0 and nvir>0:
                    H_ai[:, :, i, j] = np.einsum('pa,pi->ai', h1e[:, nocc:], dm1[:, :ncore]) + np.einsum('ap,ip->ai', h1e[nocc:, :], dm1[:ncore, :]) + np.einsum('paqr,piqr->ai', eri[:, nocc:, :, :], dm2[:, :ncore, :, :]) + np.einsum('aqpr,iqpr->ai', eri[nocc:, :, :, :], dm2[:ncore, :, :, :])
                if nvir>0:
                    H_at[:, :, i, j] = np.einsum('pa,pt->at', h1e[:, nocc:], dm1[:, ncore:nocc]) + np.einsum('ap,tp->at', h1e[nocc:, :], dm1[ncore:nocc, :]) + np.einsum('paqr,ptqr->at', eri[:, nocc:, :, :], dm2[:, ncore:nocc, :, :]) + np.einsum('aqpr,tqpr->at',eri[nocc:, :, :, :], dm2[ncore:nocc, :, :, :])
                if ncore>0:
                    H_ti[:, :, i, j] = np.einsum('pt,pi->ti', h1e[:, ncore:nocc], dm1[:, :ncore]) + np.einsum('tp,ip->ti', h1e[ncore:nocc, :], dm1[:ncore, :]) + np.einsum('ptqr,piqr->ti', eri[:, ncore:nocc, :, :], dm2[:, :ncore, :, :]) + np.einsum('tqpr,iqpr->ti', eri[ncore:nocc, :, :, :], dm2[:ncore, :, :, :]) - np.einsum('pi,pt->ti', h1e[:, :ncore], dm1[:, ncore:nocc]) - np.einsum('ip,tp->ti', h1e[:ncore, :], dm1[ncore:nocc, :]) - np.einsum('piqr,ptqr->ti', eri[:, :ncore, :, :], dm2[:, ncore:nocc, :, :]) - np.einsum('iqpr,tqpr->ti', eri[:ncore, :, :, :], dm2[ncore:nocc, :, :, :])

        return H_ai, H_at, H_ti

    def get_hessianOrbCI(self): #TODO
        ''' This method build the orb-CI part of the hessian '''
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        H_OCI = np.zeros((self.norb,self.norb,self.nDet-1))
        mat_CI = self.mat_CI

        H_ai, H_at, H_ti = self.get_hamiltonianComm()

        ci0 = mat_CI[:,0]

        h1e = self.h1e
        eri = self.eri

        for k in range(len(mat_CI)-1): # Loop on Hessian indices
                cleft = mat_CI[:,k+1]
                H_OCI[nocc:, :ncore, k] = 2*np.einsum('k,aikl,l->ai', cleft, H_ai, ci0)
                H_OCI[nocc:, ncore:nocc, k] = 2*np.einsum('k,aikl,l->ai', cleft, H_at, ci0)
                H_OCI[ncore:nocc, :ncore, k] = 2*np.einsum('k,aikl,l->ai', cleft, H_ti, ci0)

        H_OCI = H_OCI + np.einsum('pqs->qps',H_OCI)

        return H_OCI

    def get_hessian(self): #TODO
        ''' This method concatenate the orb-orb, orb-CI and CI-CI part of the Hessian '''
        norb = self.norb
        nDet = self.nDet

        idx = self.uniq_var_indices(norb,self.frozen)

        dm1_cas, dm2_cas = self.get_CASRDM_12(self.mat_CI[:,0])

        H_OrbOrb = self.get_hessianOrbOrb(dm1_cas,dm2_cas)
        H_CICI = self.get_hessianCICI()
        H_OrbCI = self.get_hessianOrbCI()

        H_OrbOrb = H_OrbOrb[:,:,idx]
        H_OrbOrb = H_OrbOrb[idx,:]
        H_OrbCI = H_OrbCI[idx,:]

        nIndepOrb = len(H_OrbOrb)
        H = np.zeros((nIndepOrb+nDet-1,nIndepOrb+nDet-1))

        H[:nIndepOrb, :nIndepOrb] = H_OrbOrb
        H[:nIndepOrb, nIndepOrb:] = H_OrbCI
        H[nIndepOrb:, :nIndepOrb] = H_OrbCI.T
        H[nIndepOrb:, nIndepOrb:] = H_CICI

        return H

    def rotateOrb(self,K):
        mo = np.dot(self.mo_coeff, scipy.linalg.expm(K))
        return mo

    def rotateCI(self,S): #TODO check the renormalization ?
        ci = np.dot(scipy.linalg.expm(S),self.mat_CI)
        #We need to renormalize the CI states
        ci = ci/np.dot(ci[:,0], ci[:,0].T)
        return ci

    def numericalGrad(self):
        epsilon = 0.00000001
        dm1_cas, dm2_cas = self.get_CASRDM_12(self.mat_CI[:,0])
        e0 = self.get_energy(self.h1e, self.eri, dm1_cas, dm2_cas)
        # Orbital gradient
        g_orb = np.zeros((self.norb,self.norb))
        for p in range(1,self.norb):
            for q in range(p):
                K = np.zeros((self.norb,self.norb))
                K[p,q] = epsilon
                K[q,p] = -epsilon
                mo_coeff = self.rotateOrb(K)
                h1eUpdate = np.einsum('ip,ij,jq->pq', mo_coeff, self.h1e_AO, mo_coeff)
                eriUpdate = np.asarray(mol.ao2mo(mo_coeff))
                eriUpdate = ao2mo.restore(1, eriUpdate, self.norb) # eri in the MO basis with chemist notation
                eUpdate = self.get_energy(h1eUpdate, eriUpdate, dm1_cas, dm2_cas)
                g_orb[p,q] = (eUpdate - e0)/epsilon
        g_orb = g_orb - g_orb.T

        idx = self.uniq_var_indices(self.norb,self.frozen)
        g_orb = self.pack_uniq_var(g_orb)

        # CI gradient
        g_CI = np.zeros(self.nDet - 1)

        for k in range(1,self.nDet):
            S = np.zeros((self.nDet,self.nDet))
            Sk0 = epsilon
            for i in range(self.nDet):
                for j in range(self.nDet):
                    S[i,j] = Sk0*(self.mat_CI[i,k]*self.mat_CI[j,0] - self.mat_CI[j,k]*self.mat_CI[i,0])
            ciUpdate = self.rotateCI(S)
            dm1_casUpdate, dm2_casUpdate = self.get_CASRDM_12(ciUpdate[:,0])
            eUpdate = self.get_energy(self.h1e, self.eri, dm1_casUpdate, dm2_casUpdate)
            g_CI[k-1] = (eUpdate - e0)/epsilon

        g = np.concatenate((g_orb,g_CI))

        return g

    def numericalHessian(self): #TODO
        epsilon = 0.00001
        dm1_cas, dm2_cas = self.get_CASRDM_12(self.mat_CI[:,0])
        e0 = self.get_energy(self.h1e, self.eri, dm1_cas, dm2_cas)
        #OrbOrbHessian
        norb = self.norb
        H_OrbOrb = np.zeros((norb,norb,norb,norb))
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        for p in range(norb):
            for q in range(norb):
                for r in range(norb):
                    for s in range(norb):
                        # Kpp = np.zeros((norb,norb))
                        # Kpp[p,q] = epsilon
                        # Kpp[q,p] = -epsilon
                        # Kpp[r,s] = epsilon
                        # Kpp[s,r] = -epsilon
                        #
                        # Kpm = np.zeros((norb,norb))
                        # Kpm[p,q] = epsilon
                        # Kpm[q,p] = -epsilon
                        # Kpm[r,s] = -epsilon
                        # Kpm[s,r] = epsilon
                        #
                        # Kmp = np.zeros((norb,norb))
                        # Kmp[p,q] = -epsilon
                        # Kmp[q,p] = epsilon
                        # Kmp[r,s] = epsilon
                        # Kmp[s,r] = -epsilon
                        #
                        # Kmm = np.zeros((norb,norb))
                        # Kmm[p,q] = -epsilon
                        # Kmm[q,p] = epsilon
                        # Kmm[r,s] = -epsilon
                        # Kmm[s,r] = epsilon
                        #
                        # mo_coeffpp = self.rotateOrb(Kpp)
                        # mo_coeffpm = self.rotateOrb(Kpm)
                        # mo_coeffmp = self.rotateOrb(Kmp)
                        # mo_coeffmm = self.rotateOrb(Kmm)
                        #
                        # h1eUpdatepp = np.einsum('ip,ij,jq->pq', mo_coeffpp, self.h1e_AO, mo_coeffpp)
                        # h1eUpdatepm = np.einsum('ip,ij,jq->pq', mo_coeffpm, self.h1e_AO, mo_coeffpm)
                        # h1eUpdatemp = np.einsum('ip,ij,jq->pq', mo_coeffmp, self.h1e_AO, mo_coeffmp)
                        # h1eUpdatemm = np.einsum('ip,ij,jq->pq', mo_coeffmm, self.h1e_AO, mo_coeffmm)
                        #
                        # eriUpdatepp = np.asarray(mol.ao2mo(mo_coeffpp))
                        # eriUpdatepp = ao2mo.restore(1, eriUpdatepp, norb)
                        # eriUpdatepm = np.asarray(mol.ao2mo(mo_coeffpm))
                        # eriUpdatepm = ao2mo.restore(1, eriUpdatepm, norb)
                        # eriUpdatemp = np.asarray(mol.ao2mo(mo_coeffmp))
                        # eriUpdatemp = ao2mo.restore(1, eriUpdatemp, norb)
                        # eriUpdatemm = np.asarray(mol.ao2mo(mo_coeffmm))
                        # eriUpdatemm = ao2mo.restore(1, eriUpdatemm, norb)
                        #
                        # eUpdatepp = self.get_energy(h1eUpdatepp, eriUpdatepp, dm1_cas, dm2_cas)
                        # eUpdatepm = self.get_energy(h1eUpdatepm, eriUpdatepm, dm1_cas, dm2_cas)
                        # eUpdatemp = self.get_energy(h1eUpdatemp, eriUpdatemp, dm1_cas, dm2_cas)
                        # eUpdatemm = self.get_energy(h1eUpdatemm, eriUpdatemm, dm1_cas, dm2_cas)
                        #
                        # H_OrbOrb[p,q,r,s] = (eUpdatepp + eUpdatemm - eUpdatepm - eUpdatemp)/(4*(epsilon**2))

                        Kpqrs = np.zeros((norb,norb))
                        Kpqrs[p,q] = epsilon
                        Kpqrs[q,p] = -epsilon
                        Kpqrs[r,s] = epsilon
                        Kpqrs[s,r] = -epsilon

                        Kpq = np.zeros((norb,norb))
                        Kpq[p,q] = epsilon
                        Kpq[q,p] = -epsilon

                        Krs = np.zeros((norb,norb))
                        Krs[r,s] = epsilon
                        Krs[s,r] = -epsilon

                        mo_coeffpqrs = self.rotateOrb(Kpqrs)
                        mo_coeffpq = self.rotateOrb(Kpq)
                        mo_coeffrs = self.rotateOrb(Krs)

                        h1eUpdatepqrs = np.einsum('ip,ij,jq->pq', mo_coeffpqrs, self.h1e_AO, mo_coeffpqrs)
                        h1eUpdatepq = np.einsum('ip,ij,jq->pq', mo_coeffpq, self.h1e_AO, mo_coeffpq)
                        h1eUpdaters = np.einsum('ip,ij,jq->pq', mo_coeffrs, self.h1e_AO, mo_coeffrs)

                        eriUpdatepqrs = np.asarray(mol.ao2mo(mo_coeffpqrs))
                        eriUpdatepqrs = ao2mo.restore(1, eriUpdatepqrs, norb)
                        eriUpdatepq = np.asarray(mol.ao2mo(mo_coeffpq))
                        eriUpdatepq = ao2mo.restore(1, eriUpdatepq, norb)
                        eriUpdaters = np.asarray(mol.ao2mo(mo_coeffrs))
                        eriUpdaters = ao2mo.restore(1, eriUpdaters, norb)

                        eUpdatepqrs = self.get_energy(h1eUpdatepqrs, eriUpdatepqrs, dm1_cas, dm2_cas)
                        eUpdatepq = self.get_energy(h1eUpdatepq, eriUpdatepq, dm1_cas, dm2_cas)
                        eUpdaters = self.get_energy(h1eUpdaters, eriUpdaters, dm1_cas, dm2_cas)

                        H_OrbOrb[p,q,r,s] = (eUpdatepqrs + e0 - eUpdatepq - eUpdaters)/(epsilon**2)


        # print(H_OrbOrb)
        idx = self.uniq_var_indices(self.norb,self.frozen)
        idxidx = np.einsum('pq,rs->pqrs',idx,idx)
        # print(H_OrbOrb[idxidx])
        H_OrbOrb = H_OrbOrb[:,:,idx]
        H_OrbOrb = H_OrbOrb[idx,:]

        #CICIHessian
        H_CICI = np.zeros((self.nDet - 1,self.nDet - 1))
        for k in range(1,self.nDet):
            for l in range(1,self.nDet):
                Spp = np.zeros((self.nDet,self.nDet))
                Spm = np.zeros((self.nDet,self.nDet))
                Smp = np.zeros((self.nDet,self.nDet))
                Smm = np.zeros((self.nDet,self.nDet))

                for i in range(self.nDet):
                    for j in range(self.nDet):
                        Spp[i,j] = epsilon*(self.mat_CI[i,k]*self.mat_CI[j,0] - self.mat_CI[j,k]*self.mat_CI[i,0]) + epsilon*(self.mat_CI[i,l]*self.mat_CI[j,0] - self.mat_CI[j,l]*self.mat_CI[i,0])
                        Spm[i,j] = epsilon*(self.mat_CI[i,k]*self.mat_CI[j,0] - self.mat_CI[j,k]*self.mat_CI[i,0]) - epsilon*(self.mat_CI[i,l]*self.mat_CI[j,0] - self.mat_CI[j,l]*self.mat_CI[i,0])
                        Smp[i,j] = - epsilon*(self.mat_CI[i,k]*self.mat_CI[j,0] - self.mat_CI[j,k]*self.mat_CI[i,0]) + epsilon*(self.mat_CI[i,l]*self.mat_CI[j,0] - self.mat_CI[j,l]*self.mat_CI[i,0])
                        Smm[i,j] = - epsilon*(self.mat_CI[i,k]*self.mat_CI[j,0] - self.mat_CI[j,k]*self.mat_CI[i,0]) - epsilon*(self.mat_CI[i,l]*self.mat_CI[j,0] - self.mat_CI[j,l]*self.mat_CI[i,0])

                ciUpdatepp = self.rotateCI(Spp)
                ciUpdatepm = self.rotateCI(Spm)
                ciUpdatemp = self.rotateCI(Smp)
                ciUpdatemm = self.rotateCI(Smm)

                dm1_casUpdatepp, dm2_casUpdatepp = self.get_CASRDM_12(ciUpdatepp[:,0])
                dm1_casUpdatepm, dm2_casUpdatepm = self.get_CASRDM_12(ciUpdatepm[:,0])
                dm1_casUpdatemp, dm2_casUpdatemp = self.get_CASRDM_12(ciUpdatemp[:,0])
                dm1_casUpdatemm, dm2_casUpdatemm = self.get_CASRDM_12(ciUpdatemm[:,0])

                eUpdatepp = self.get_energy(self.h1e, self.eri, dm1_casUpdatepp, dm2_casUpdatepp)
                eUpdatepm = self.get_energy(self.h1e, self.eri, dm1_casUpdatepm, dm2_casUpdatepm)
                eUpdatemp = self.get_energy(self.h1e, self.eri, dm1_casUpdatemp, dm2_casUpdatemp)
                eUpdatemm = self.get_energy(self.h1e, self.eri, dm1_casUpdatemm, dm2_casUpdatemm)

                H_CICI[k-1,l-1] = (eUpdatepp + eUpdatemm - eUpdatepm - eUpdatemp)/(4*(epsilon**2))

        #OrbCIHessian
        H_OrbCI = np.zeros((self.norb,self.norb,self.nDet - 1))
        for p in range(norb):
            for q in range(norb):
                Kp = np.zeros((norb,norb))
                Kp[p,q] = epsilon
                Kp[q,p] = -epsilon

                Km = np.zeros((norb,norb))
                Km[p,q] = -epsilon
                Km[q,p] = epsilon

                mo_coeffp = self.rotateOrb(Kp)
                mo_coeffm = self.rotateOrb(Km)

                h1eUpdatep = np.einsum('ip,ij,jq->pq', mo_coeffp, self.h1e_AO, mo_coeffp)
                h1eUpdatem = np.einsum('ip,ij,jq->pq', mo_coeffm, self.h1e_AO, mo_coeffm)

                eriUpdatep = np.asarray(mol.ao2mo(mo_coeffp))
                eriUpdatep = ao2mo.restore(1, eriUpdatep, norb)
                eriUpdatem = np.asarray(mol.ao2mo(mo_coeffm))
                eriUpdatem = ao2mo.restore(1, eriUpdatem, norb)
                for k in range(1,self.nDet):
                    Sp = np.zeros((self.nDet,self.nDet))
                    Sm = np.zeros((self.nDet,self.nDet))
                    for i in range(self.nDet):
                        for j in range(self.nDet):
                                    Sp[i,j] = epsilon*(self.mat_CI[i,k]*self.mat_CI[j,0] - self.mat_CI[j,k]*self.mat_CI[i,0])
                                    Sm[i,j] = -epsilon*(self.mat_CI[i,k]*self.mat_CI[j,0] - self.mat_CI[j,k]*self.mat_CI[i,0])
                    ciUpdatep = self.rotateCI(Sp)
                    ciUpdatem = self.rotateCI(Sm)

                    dm1_casUpdatep, dm2_casUpdatep = self.get_CASRDM_12(ciUpdatep[:,0])
                    dm1_casUpdatem, dm2_casUpdatem = self.get_CASRDM_12(ciUpdatem[:,0])

                    eUpdatepp = self.get_energy(h1eUpdatep, eriUpdatep, dm1_casUpdatep, dm2_casUpdatep)
                    eUpdatepm = self.get_energy(h1eUpdatep, eriUpdatep, dm1_casUpdatem, dm2_casUpdatem)
                    eUpdatemp = self.get_energy(h1eUpdatem, eriUpdatem, dm1_casUpdatep, dm2_casUpdatep)
                    eUpdatemm = self.get_energy(h1eUpdatem, eriUpdatem, dm1_casUpdatem, dm2_casUpdatem)

                    H_OrbCI[p,q,k-1] = (eUpdatepp + eUpdatemm - eUpdatepm - eUpdatemp)/(4*(epsilon**2))

        return H_OrbOrb, H_CICI, H_OrbCI[idx,:]

    def get_energy(self, h1e, eri, dm1_cas, dm2_cas):
        dm1 = self.CASRDM1_to_RDM1(dm1_cas)
        dm2 = self.CASRDM2_to_RDM2(dm1_cas,dm2_cas)
        E = np.einsum('pq,pq', h1e, dm1) + 0.5*np.einsum('pqrs,pqrs', eri, dm2)
        return E

    def kernel(): #TODO
        ''' This method runs the iterative Newton-Raphson loop '''
        pass

##### Main #####
if __name__ == '__main__':

    def matprint(mat, fmt="g"):
        if len(np.shape(np.asarray(mat)))==1:
            mat = mat.reshape(1,len(mat))

        col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
            print("")

    def delta_kron(i,j):
        if i==j:
            return 1
        else:
            return 0

    def test_run(mycas):

        # print("The MOs haven't been initialized",mycas.mo_coeff)
        mycas.initMO
        print("InitMO")
        matprint(mycas.initMO)
        print("")
        # print("The MOs are now equal to their init value",mycas.mo_coeff)
        mycas.initCI
        print("InitCI")
        matprint(mycas.initCI)
        print("")
        # print("The CI matrix is now equal to its init value",mycas.mat_CI)

        mycas.check_sanity()

        # print("We compute the 4 init 1CASRDM corresponding to the four CI vectors in mycas.mat_CI")
        # print(mycas.get_CASRDM_1(mycas.mat_CI[:,0]))
        # print(mycas.get_CASRDM_1(mycas.mat_CI[:,1]))
        # print(mycas.get_CASRDM_1(mycas.mat_CI[:,2]))
        # print(mycas.get_CASRDM_1(mycas.mat_CI[:,3]))

        dm1_cas, dm2_cas = mycas.get_CASRDM_12(mycas.mat_CI[:,0])

        # print("One init 2CASRDM",dm2_cas)
        # print("Transform the 1CASRDM to 1RDM", mycas.CASRDM1_to_RDM1(mycas.get_CASRDM_1(mycas.mat_CI[:,0])))
        # print("2CASRDM transformed into 2RDM",mycas.CASRDM2_to_RDM2(dm1_cas, dm2_cas))

        # t_dm1 = mycas.get_tCASRDM1(mycas.mat_CI[:,0],mycas.mat_CI[:,0])
        # print("Two 1el transition density matrices",t_dm1)
        # t_dm1 = mycas.get_tCASRDM1(mycas.mat_CI[:,0],mycas.mat_CI[:,1])
        # print(t_dm1)
        # t_dm1, t_dm2 = mycas.get_tCASRDM12(mycas.mat_CI[:,0],mycas.mat_CI[:,1])
        # print("A 2TDM",t_dm2)

        # print("Generalized Fock operator")
        # matprint(mycas.get_genFock(dm1_cas))
        # print("")
        g_orb = mycas.get_gradOrb(dm1_cas, dm2_cas)
        # print("Orbital gradient")
        # matprint(g_orb)
        # print("")
        # print("This is the exponential of the gradient")
        # matprint(scipy.linalg.expm(g_orb))
        # print("")

        g_ci = mycas.get_gradCI()
        # print("CI gradient")
        # matprint(g_ci)
        # print("")

        print("Full gradient")
        AlgGrad = mycas.form_grad(g_orb,g_ci)
        matprint(AlgGrad)
        print("")
        print('This is the numerical gradient')
        NumGrad = mycas.numericalGrad()
        matprint(NumGrad)
        print("")

        print("Is the numerical gradient equal to the algebraic one?", np.allclose(AlgGrad,NumGrad,atol=1e-06))

        print("This is the Hamiltonian")
        # matprint(mycas.get_hamiltonian())
        print("")
        print("This is the CICI hessian")
        # matprint(mycas.get_hessianCICI())
        print("")

        # H_OO = mycas.get_hessianOrbOrb(dm1_cas,dm2_cas)
        # print("This is the OrbOrb hessian", H_OO)
        # idx = mycas.uniq_var_indices(mycas.norb,mycas.frozen)
        # print("This is the mask of uniq orbitals";idx)
        # tmp = H_OO[:,:,idx]
        # tmp = tmp[idx,:]
        # print("This is the hessian of independant rotations",tmp)

        # print('This are the Hamiltonian commutators', mycas.get_hamiltonianComm())
        # print("This is the OrbCI hessian", mycas.get_hessianOrbCI())
        # print('This is the OrbCI hessian with unique orbital rotations', mycas.get_hessianOrbCI()[idx,:])

        print('This is the hessian')
        # matprint(mycas.get_hessian())
        print("")

        numOO, numCICI, numOCI = mycas.numericalHessian()

        # print('This is the CICI numerical Hessian')
        # matprint(numCICI)
        # print("")
        # print("This is the CICI hessian")
        # matprint(mycas.get_hessianCICI())
        # print("")

        print('This is the orborb numerical Hessian')
        matprint(numOO)
        print('This is the orborb hessian')
        matprint(mycas.get_hessian()[:mycas.norb,:mycas.norb])

        # ncore = mycas.ncore
        # nocc = mycas.ncas + ncore
        # print(mycas.eri[:, :, ncore:nocc, ncore:nocc])

        # mycas.get_hessian()
        # print(numOCI)

        return

    # mol = pyscf.M(
    #     atom = 'H 0 0 0; H 0 0 1.05',
    #     basis = 'sto-3g')
    # myhf = mol.RHF().run()
    # mycas = NR_CASSCF(myhf,2,2)
    #
    # test_run(mycas)
    #
    # mol = pyscf.M(
    #     atom = 'H 0 0 0; H 0 0 1.05',
    #     basis = 'sto-3g')
    # myhf = mol.RHF().run()
    # mat = np.asarray([[1/np.sqrt(2),1/np.sqrt(2),0,0],[1/np.sqrt(2),-1/np.sqrt(2),0,0],[0,0,1,0],[0,0,0,1]])
    # mycas = NR_CASSCF(myhf,2,2,initCI=mat)
    #
    # test_run(mycas)

    mol = pyscf.M(
        atom = 'H 0 0 0; H 0 0 1.2',
        basis = '6-31g')
    myhf = mol.RHF().run()
    mycas = NR_CASSCF(myhf,2,2)

    test_run(mycas)

    # mol = pyscf.M(
    #     atom = 'H 0 0 0; H 0 0 1.2',
    #     basis = '6-31g')
    # myhf = mol.RHF().run()
    # mat = np.asarray([[1/np.sqrt(2),1/np.sqrt(2),0,0],[1/np.sqrt(2),-1/np.sqrt(2),0,0],[0,0,1,0],[0,0,0,1]])
    # mycas = NR_CASSCF(myhf,2,2,initCI=mat)
    #
    # test_run(mycas)

    # In the two sto-3g cases, the orbital gradient is equal to zero. This is expected because the orbitals are determined by the symmetry of the system.
    # In the first 6-31g case, the orbital gradient is equal to zero because the wave function is the ground state determinant. In the second case, the orbital gradient is non-zero.
    # If we run several times this script we may obtain different results, i.e. the molecular orbitals of H2 are determined up to a sign.
    # + -  is equivalent to - +
    # + +                   + +
    # It changes the sign of some terms of the orbital gradient.

    # The CICI and OrbCI hessians are in agreement with their numerical counterpart. Still need to work on the OrbOrb part ...

    # When OrbOrb will be fixed, I need to test the code with ncore>0.




    #TODO
    # add this code
    # h1 = myhf.mo_coeff.T.dot(myhf.get_hcore()).dot(myhf.mo_coeff)
    # h2 = ao2mo.kernel(mol,myhf.mo_coeff)
    # h2 = ao2mo.restore(1,h2,mol.nao_nr())
    # H_fci=fci.direct_spin1.pspace(h1,h2,mol.nao_nr(),mol.nelec,np=10000)[1]
    #
    # matprint(H_fci)


