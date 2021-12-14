#!/usr/bin/env python
# Author: Antoine Marie

import sys

from functools import reduce
import numpy as np
import scipy.special
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
        if isinstance(myhf_or_mol, gto.Mole): # Check if the arg is an HF object or a molecule object
            myhf = scf.RHF(myhf_or_mol)
        else:
            myhf = myhf_or_mol

        mol = myhf.mol
        self.mol = mol
        self.nelec = mol.nelec #Access the number of electrons
        self._scf = myhf
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = myhf.max_memory
        self.ncas = ncas
        if isinstance(nelecas, (int, np.integer)):
            nelecb = (nelecas-mol.spin)//2
            neleca = nelecas - nelecb
            self.nelecas = (neleca, nelecb)
        else:
            self.nelecas = (nelecas[0],nelecas[1]).astype(int)
        self.v1e = mol.intor('int1e_nuc') #Nuclear repulsion matrix elements
        self.t1e = mol.intor('int1e_kin') #Kinetic energy matrix elements
        self.h1e_AO =  self.t1e + self.v1e
        self.norb = len(self.h1e_AO)
        self.nDeta = (scipy.special.comb(self.ncas,self.nelecas[0])).astype(int)
        self.nDetb = (scipy.special.comb(self.ncas,self.nelecas[1])).astype(int)
        self.nDet = (self.nDeta*self.nDetb).astype(int)
        self.eri_AO = mol.intor('int2e') #eri in the AO basis in the chemist notation
        self._ncore = ncore
        self._initMO = initMO
        self._initCI = initCI
        self.frozen = frozen
        self.mo_coeff = None
        self.mat_CI = None

        self.fcisolver = fci.direct_spin1.FCISolver(mol)

    @property
    def ncore(self):
        if self._ncore is None:
            ncorelec = self.mol.nelectron - sum(self.nelecas)
            assert ncorelec % 2 == 0
            assert ncorelec >= 0
            return ncorelec // 2
        else:
            return self._ncore

    @property
    def initMO(self):
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
        if self._initCI is None:
            self.mat_CI = np.identity(self.nDet, dtype="float")
            return np.identity(self.nDet, dtype="float")
        else:
            self.mat_CI = initCI
            return self._initCI

    def dump_flags(self): #TODO
        '''
        '''
        pass

    def check_sanity(self): #TODO
        '''
        '''
        pass

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
            dm1_core = 2*np.identity(self.ncore, dtype="int") #the occocc part of dm1 is a diagonal of 2
            dm1[:ncore,:ncore] = dm1_core
        dm1[ncore:ncore+ncas,ncore:ncore+ncas] = dm1_cas
        return dm1

    @staticmethod
    def dm2_mo_occ(part,p,q,r,s,dm1):
        ''' This function compute the core part of the 2-RDM. The ijkl, ijpq and pijq can be simplified according to the following equations (Eq(62)-(67)). The piqj is zero as well as all the elements with a virtual index or an odd number of core indices. '''
        if part=='ijkl':
            return(2*np.kron(p,q)*np.kron(r,s) - np.kron(p,r)*np.kron(q,s))
        elif part=='ijpq':
            return(np.kron(p,q)*dm1[r,s] - np.kron(p,s)*np.kron(q,r))
        elif part=='pijq':
            return(2*np.kron(q,p)*np.kron(r,s) - 0.5*np.kron(q,r)*dm1[p,q])
        else:
            return('Wrong specification of part')

    def CASRDM2_to_RDM2(self,dm1_cas,dm2_cas):
        ''' This method takes a 2-RDM in the CAS space and transform it to the full MO space '''
        ncore = self.ncore
        ncas = self.ncas
        norb = self.norb
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

        dm2[ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas] = dm2_cas # Finally we add the uvxy sector
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

    @staticmethod
    def get_F_core(norb,ncore,h1e,eri): #TODO we could probably optimize this...
        F_core = np.zeros((norb,norb))
        F_core += h1e
        for p in range(norb):
            for q in range(norb):
                for i in range(ncore):
                    F_core[p,q] += 2*eri[p,q,i,i] - eri[p,i,i,q]
        return(F_core)

    @staticmethod
    def get_F_cas(norb,ncas,dm1_cas,eri): #TODO same as above
        F_cas = np.zeros((norb,norb))
        for p in range(norb):
            for q in range(norb):
                for t in range(ncas):
                    for u in range(ncas):
                        F_cas[p,q] += dm1_cas[t,u]*(eri[p,q,t,u]-0.5*eri[p,u,t,q])
        return(F_cas)

    def get_genFock(self,dm1_cas):
        ''' This method build the generalized Fock matrix '''
        return self.get_F_core(self.norb,self.ncore,self.h1e,self.eri) + self.get_F_cas(self.norb,self.ncas,dm1_cas,self.eri)

    def get_gradOrb(self,dm1_cas,dm2_cas):
        ''' This method build the orbital part of the gradient '''
        g_orb = np.zeros((self.norb,self.norb))
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        print(nocc)
        nvir = self.norb - nocc
        F_core = self.get_F_core(self.norb,ncore,self.h1e,self.eri)
        F_cas = self.get_F_cas(self.norb,ncas,dm1_cas,self.eri)

        #virtual-core rotations g_{ai}
        if ncore>0 and nvir>0:
            g_orb[nocc:,:ncore] = 4*(F_core + F_cas)[nocc:,:ncore]
        #active-core rotations g_{ti}
        if ncore>0:
            g_orb[ncore:nocc,:ncore] = 4*(F_core + F_cas)[ncore:nocc,:ncore] - 2*np.einsum('tv,iv->ti',dm1_cas,F_core[:ncore,ncore:nocc]) - 4*np.einsum('tvxy,ivxy->ti',dm2_cas,self.eri[:ncore,ncore:nocc,ncore:nocc,ncore:nocc])
        #virtual-active rotations g_{at}
        if nvir>0:
            g_orb[nocc:,ncore:nocc] = 2*np.einsum('tv,av->at',dm1_cas,F_core[nocc:,ncore:nocc]) + 4*np.einsum('tvxy,avxy->at',dm2_cas,self.eri[nocc:,ncore:nocc,ncore:nocc,ncore:nocc])
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

    def get_hessianOrbOrb(self,dm1_cas,dm2_cas): #TODO optimize
        ''' This method build the orb-orb part of the hessian '''
        norb = self.norb
        H = np.zeros((norb,norb,norb,norb))
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        nvir = norb - nocc
        eri = self.eri
        F_core = self.get_F_core(norb,ncore,self.h1e,self.eri)
        F_cas = self.get_F_cas(norb,ncas,dm1_cas,self.eri)
        F_tot = F_core + F_cas

        #virtual-core virtual-core H_{ai,bj}
        if ncore>0 and nvir>0:
            tmp = eri[nocc:, :ncore, nocc:, :ncore]
            H[nocc:, :ncore, nocc:, :ncore] = 2(4*tmp - np.einsum('aibj->abij',tmp) -np.einsum('aibj->ajib',tmp)) + 2*np.einsum('ij,ab->aibj',np.identity(ncore),F_tot[nocc:,nocc:]) - 2*np.einsum('ab,ij->aibj', np.identity(nvir), F_tot[:ncore,:ncore])


        #virtual-core active-core H_{ai,tj}
        if ncore>0 and nvir>0:
            tmp = eri[nocc:, :ncore, ncore:nocc, :ncore]
            eri_tmp = (4*tmp - np.einsum('aibv->avbi',tmp) -np.einsum('aibv->abvi',tmp))
            H[nocc:, :ncore, ncore:nocc, :ncore] = np.sum('tv,aibv->aibt', cas_dm1, eri_tmp) - np.einsum('ab,tvxyv,vixy ->aibt', np.identity(nvir), dm2_cas,eri[ncore:nocc, :ncore, ncore:nocc, ncore:nocc]) - np.einsum('ab,ti->aibt', np.identity(nvir), F_tot[ncore:nocc, :ncore]) - 0.5*np.einsum('ab,tv,vi->aibt', np.identity(nvir), dm1_cas, F_core[ncore:nocc, :ncore])

        #virtual-core virtual-active H_{ai,bt}
        if ncore>0 and nvir>0:
            tmp = eri[nocc:, :ncore, nocc:, ncore:nocc]
            eri_tmp = (4*tmp - np.einsum('aivj->avji',tmp) -np.einsum('aivj->ajvi',tmp))
            H[nocc:, :ncore, nocc:, ncore:nocc] = np.sum('tv,aibv->aibt', (2*np.identity(ncas) - cas_dm1), eri_tmp) - np.einsum('ji,tvxyv,avxy -> aitj', np.identity(ncore), dm2_cas,eri[nocc:, ncore:nocc, ncore:nocc, ncore:nocc]) + 2*np.einsum('ij,at-> aitj', np.identity(ncore), F_tot[nocc:, ncore:nocc]) - 0.5*np.einsum('ij,tv,av-> aitj', np.identity(ncore), dm1_cas, F_core[nocc:, ncore:nocc])

        #active-core active-core H_{ti,uj}
        if ncore>0:
            tmp1 = 2*np.einsum('utvx,vxij->tiuj', dm2_cas, eri[ncore:nocc, ncore:nocc, :ncore, :ncore]) + 2*np.einsum('uxvt,vxij->tiuj', dm2_cas, eri[ncore:nocc, ncore:nocc, :ncore, :ncore]) + 2*np.einsum('uxtv,vxij->tiuj', dm2_cas, eri[ncore:nocc, :ncore, ncore:nocc, :ncore])

            tmp = eri[ncore:nocc, :ncore, ncore:nocc, :ncore]
            eri_tmp1 = (4*tmp - np.einsum('viuj->uivj',tmp) -np.einsum('viuj->uvij',tmp))
            eri_tmp2 = (4*tmp - np.einsum('vjti->tjvi',tmp) -np.einsum('vjti->tvij',tmp))
            tmp2 = np.einsum('tv,viuj->tiuj', np.identity(ncas) - cas_dm1, eri_tmp1) + np.einsum('uv,vjti->tiuj', np.identity(ncas) - cas_dm1, eri_tmp2)

            tmp3 = np.einsum('tu,ij->tiuj', cas_dm1, F_core[:ncore,:ncore]) - 2*np.einsum('ij,tvxy,uvxy->tiuj', np.identity(ncore), cas_dm2, eri[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc]) - np.einsum('ij,uv,tv->tiuj', np.identity(ncore), cas_dm1, F_core[ncore:nocc, ncore:nocc]) + 2*np.einsum('ij,tu->tiuj', np.identity(ncore), F_tot[ncore:nocc, ncore:nocc]) - 2*np.einsum('tu,ij->tiuj', np.identity(ncas), F_tot[:ncore,:ncore])

            H[ncore:nocc, :ncore, ncore:nocc, :ncore] = tmp1 + tmp2 + tmp3

        #active-core virtual-active H_{ti,au}
        if ncore>0 and nvir>0:
            tmp1 = - 2*np.einsum('tuvx,aivx->tiau', dm2_cas, eri[nocc:, :ncore, ncore:nocc, ncore:nocc]) + 2*np.einsum('tvux,axvi->tiau', dm2_cas, eri[nocc:, ncore:nocc, ncore:nocc, :ncore]) + 2*np.einsum('tvxu,axvi->tiau', dm2_cas, eri[nocc:, ncore:nocc, ncore:nocc, :ncore])

            tmp = eri[nocc:, ncore:nocc, ncore:nocc, :ncore]
            eri_tmp = (4*tmp - np.einsum('avti->aitv',tmp) -np.einsum('avti->atvi',tmp))
            tmp2 = np.einsum('uv,aitv->tiau', dm1_cas, eri_tmp) - np.einsum('tu,ai->tiau', dm1_cas, F_core[nocc:, :ncore]) + np.einsum('tu,ai->tiau',np.identity(ncas),F_tot[nocc:,:ncore])

            H[ncore:nocc, :ncore, nocc:, ncore:nocc] = tmp1 + tmp2

        #virtual-active virtual-active H_{at,bu}
        if nvir>0:
            tmp1 = 2*np.einsum('tuvx,abvx->atbu', dm2_cas, eri[nocc:, nocc:, ncore:nocc, ncore:nocc]) + 2*np.einsum('tvux,axvb->atbu', dm2_cas, eri[nocc:, ncore:nocc, nocc:, ncore:nocc]) + 2*np.einsum('tvxu,axvb->atbu', dm2_cas, eri[nocc:, ncore:nocc, nocc:, ncore:nocc])

            tmp2 = - np.einsum('ab,tvxy,uvxy->atbu',np.identity(nvir),dm2_cas,eri[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc]) - 0.5*np.einsum('ab,tv,uv->atbu',np.identity(nvir),dm1_cas,F_core[ncore:nocc,nocc:])

            H[nocc:, ncore:nocc, nocc:, ncore:nocc] = tmp1 + tmp2 + np.einsum('atbu->aubt',tmp2) + np.einsum('tu,ab->atbu', dm1_cas, F_core[nocc:, nocc:])

        # anti-symmetrize the Hessian
        #TODO not sure about this

        H = H + np.einsum('pqrs->rspq',H) + np.einsum('pqrs->qpsr',H) + np.einsum('pqrs->srqp',H)

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
                H[i,j] = np.einsum('pq,pq',self.h1e,dm1) + np.einsum('pqrs,pqrs',self.eri,dm2)
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

        for k in range(len(mat_CI)-1): # Loop on Hessian indices
                cleft = mat_CI[:,k+1]
                for l in range(len(mat_CI)-1):
                    cright = mat_CI[:,l+1]
                    hessian_CICI[k,l] = 2*np.einsum('i,ij,j',cleft,H,cright) - np.kron(k,l)*eO

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

        H = np.zeros((norb+nDet-1,norb+nDet-1))

        idx = self.uniq_var_indices(norb,self.frozen)

        dm1_cas, dm2_cas = self.get_CASRDM_12(self.mat_CI[:,0])

        H_OrbOrb = self.get_hessianOrbOrb(dm1_cas,dm2_cas)
        H_CICI = self.get_hessianCICI()
        H_OrbCI = self.get_hessianOrbCI()

        H_OrbOrb = H_OrbOrb[:,:,idx]
        H_OrbOrb = H_OrbOrb[idx,:]
        H_OrbCI = H_OrbCI[idx,:]

        H[:norb, :norb] = H_OrbOrb
        H[:norb, norb:] = H_OrbCI
        H[norb:, :norb] = H_OrbCI.T
        H[norb:, norb:] = H_CICI

        return H

    def kernel(): #TODO
        ''' This method runs the iterative Newton-Raphson loop '''
        pass

##### Main #####
if __name__ == '__main__':

    mol = pyscf.M(
        atom = 'H 0 0 0; H 0 0 1.2',
        basis = '6-31g')


    # mol = pyscf.M(
    #     atom = 'H 0 0 0; H 0 0 1.889',
    #     basis = 'sto-3g')

    myhf = mol.RHF().run()

    mycas = NR_CASSCF(myhf,2,2)
    print('This is the number of core orbitals automatically calculated',mycas.ncore)

    mycas = NR_CASSCF(myhf,2,2,0)
    print('This is the number of core orbitals set by the user',mycas.ncore)

    print("The MOs haven't been initialized",mycas.mo_coeff)
    print("InitMO",mycas.initMO)
    print("The MOs are now equal to their init value",mycas.mo_coeff)
    print("InitCI",mycas.initCI)
    print("The CI matrix is now equal to its init value",mycas.mat_CI)

    print("We compute the 4 init 1CASRDM",mycas.get_CASRDM_1(mycas.mat_CI[:,0]))
    print(mycas.get_CASRDM_1(mycas.mat_CI[:,1]))
    print(mycas.get_CASRDM_1(mycas.mat_CI[:,2]))
    print(mycas.get_CASRDM_1(mycas.mat_CI[:,3]))

    print("Transform the 1CASRDM to 1RDM",mycas.CASRDM1_to_RDM1(mycas.get_CASRDM_1(mycas.mat_CI[:,0])))

    dm1_cas, dm2_cas = mycas.get_CASRDM_12(mycas.mat_CI[:,0])
    print("One init 2CASRDM",dm2_cas)

    print("2CASRDM transformed into 2RDM",mycas.CASRDM2_to_RDM2(dm1_cas, dm2_cas))

    t_dm1 = mycas.get_tCASRDM1(mycas.mat_CI[:,0],mycas.mat_CI[:,0])
    print("Two 1el transition density matrices",t_dm1)

    t_dm1 = mycas.get_tCASRDM1(mycas.mat_CI[:,0],mycas.mat_CI[:,1])
    print(t_dm1)

    t_dm1, t_dm2 = mycas.get_tCASRDM12(mycas.mat_CI[:,0],mycas.mat_CI[:,1])
    print("A 2TDM",t_dm2)

    print("Generalized Fock operator",mycas.get_genFock(dm1_cas))

    g_orb = mycas.get_gradOrb(dm1_cas,dm2_cas)
    print("Orbital gradient",g_orb)

    g_ci = mycas.get_gradCI()
    print("CI gradient",g_ci)

    print("Full gradient",mycas.form_grad(g_orb,g_ci))

    print("This is the Hamiltonian", mycas.get_hamiltonian())

    print("This is the CICI hessian", mycas.get_hessianCICI())

    H_OO = mycas.get_hessianOrbOrb(dm1_cas,dm2_cas)
    print("This is the OrbOrb hessian", H_OO)

    idx = mycas.uniq_var_indices(mycas.norb,mycas.frozen)
    print(idx)

    tmp = H_OO[:,:,idx]
    tmp = tmp[idx,:]
    print(tmp)

    print('This are the Hamiltonian commutators', mycas.get_hamiltonianComm())

    print("This is the OrbCI hessian", mycas.get_hessianOrbCI())

    print('This is the OrbCI hessian with unique orbital rotations', mycas.get_hessianOrbCI()[idx,:])

    print('This is the hessian', mycas.get_hessian())