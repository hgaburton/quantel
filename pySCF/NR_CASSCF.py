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

##### Definition of the functions #####
# If a function has no documentation, it means the documentation is included in the method with the same name below.

def get_CASRDM_12(self,ci,ncas,nelecas):

    if len(ci.shape)==1:
        nDeta = scipy.special.comb(ncas,nelecas[0])
        nDetb = scipy.special.comb(ncas,nelecas[1])
        ci = ci.reshape((nDeta,nDetb)) #TODO be careful that the reshape is the inverse of the "unfolding" of the matrix vector

    dm1_cas = self.fcisolver.make_rdm1(ci,ncas,nelecas) # the make_rdm1 method takes a ci vector as a matrix na x nb

    return dm1_cas.T # We transpose the 1-RDM because their convention is <|a_q^\dagger a_p|>

def get_CASRDM_12(self,ci,ncas,nelecas):

    if len(ci.shape)==1:
        nDeta = scipy.special.comb(ncas,nelecas[0])
        nDetb = scipy.special.comb(ncas,nelecas[1])
        ci = ci.reshape((nDeta,nDetb)) #TODO be careful that the reshape is the inverse of the "unfolding" of the matrix vector

    dm1_cas, dm2_cas = self.fcisolver.make_rdm12(ci,ncas,nelecas) # The make_rdm12 method takes a ci vector as a matrix na x nb

    return dm1_cas.T, dm2_cas # We transpose the 1-RDM because their convention is <|a_q^\dagger a_p|>

def CASRDM1_to_RDM1(casdm1,norb,ncore,ncas):
    dm1 = np.zeros((norb,norb))
    dm1_core = 2*np.diag(np.ones(ncore)) #the occocc part of dm1 is a diagonal of 2
    dm1[:ncore,:ncore] = dm1_core
    dm1[ncore:ncore+ncas,ncore:ncore+ncas] = dm1_cas
    return dm1

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

def CASRDM2_to_RDM2(dm1_cas,dm2_cas,norb,ncore,ncas):
    dm2 = np.zeros((norb,norb,norb,norb))
    for i in range(ncore):      # really not elegant ... find an alternative way to do this
        for j in range(ncore):
            for k in range(ncore):
                for l in range(ncore):
                    dm2[i,j,k,l]=dm2_mo_occ('ijkl',i,j,k,l,dm1_cas)
    for i in range(ncore):
        for j in range(ncore):
            for p in range(ncas):
                for q in range(ncas):
                    dm2[i,j,p+ncore,q+ncore]=dm2_mo_occ('ijpq',i,j,p,q,dm1_cas)
    for i in range(ncore):
        for j in range(ncore):
            for p in range(ncas):
                for q in range(ncas):
                    dm2[p+ncore,i,j,q+ncore]=dm2_mo_occ('pijq',p,i,j,q,dm1_cas)

    t_dm2[ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas] = dm2_cas # Finally we add the uvxy sector
    return dm1

def get_tCASRDM1(self,ci1,ci2,ncas,nelecas):

    if len(ci1.shape)==1:
        nDeta = scipy.special.comb(ncas,nelecas[0])
        nDetb = scipy.special.comb(ncas,nelecas[1])
        ci1 = ci1.reshape((nDeta,nDetb)) #TODO be careful that the reshape is the inverse of the "unfolding" of the matrix vector
    if len(ci2.shape)==1:
        nDeta = scipy.special.comb(ncas,nelecas[0])
        nDetb = scipy.special.comb(ncas,nelecas[1])
        ci2 = ci2.reshape((nDeta,nDetb)) #TODO be careful that the reshape is the inverse of the "unfolding" of the matrix vector

    t_dm1_cas = mycas.fcisolver.trans_rdm1(ci1,ci2,ncas,nelecas)

    return t_dm1_cas #TODO do we need to transpose this matrix

def get_tCASRDM12(self,ci1,ci2,ncas,nelecas):

    if len(ci1.shape)==1:
        nDeta = scipy.special.comb(ncas,nelecas[0])
        nDetb = scipy.special.comb(ncas,nelecas[1])
        ci1 = ci1.reshape((nDeta,nDetb)) #TODO be careful that the reshape is the inverse of the "unfolding" of the matrix vector
    if len(ci2.shape)==1:
        nDeta = scipy.special.comb(ncas,nelecas[0])
        nDetb = scipy.special.comb(ncas,nelecas[1])
        ci2 = ci2.reshape((nDeta,nDetb)) #TODO be careful that the reshape is the inverse of the "unfolding" of the matrix vector
    t_dm1_cas, t_dm2_cas = mycas.fcisolver.trans_rdm12(ci1,ci2,ncas,nelecas)

    return t_dm1_cas, t_dm2_cas #TODO do we need to transpose this matrix

def get_F_core(norb,ncore,h1e,eri): #TODO we could probably optimize this...
    F_core = np.zeros((norb,norb))
    F_core += h1e
    for p in range(norb):
        for q in range(norb):
            for i in range(ncore):
                F_core[p,q] += 2*eri[p,q,i,i] - eri[p,i,i,q]
    return(F_core)

def get_F_cas(norb,ncas,dm1_cas,eri): #TODO same as above
    F_cas = np.zeros((norb,norb))
    for p in range(norb):
        for q in range(norb):
            for t in range(ncas):
                for u in range(ncas):
                    F_cas[p,q] += dm1_cas[t,u]*(eri[p,q,t,u]-0.5*eri[p,u,t,q])
    return(F_cas)

def get_GradOrb(self,F_core,F_cas,dm1_cas,dm2_cas):
    g_orb = np.zeros((self.norb,self.norb))
    ncore = self.ncore
    ncas = self.ncas
    nocc = ncore + ncas

    g_orb[nocc:,:ncore] = 4*(F_core + F_cas)[nocc:,:ncore]
    g_orb[nocc:,ncore:nocc] = 4*(F_core + F_cas)[nocc:,ncore:nocc] - 2*np.einsum('tv,iv->ti',dm1_cas,F_core[nocc:,ncore:nocc]) - 4*np.einsum('tvxy,ivxy->ti',dm2_cas,eri)
    g_orb[ncore:nocc,:ncore] = 2*np.einsum('tv,av->ta',dm1_cas,F_core[ncore:nocc,:ncore]) + 4*np.einsum('tvxy,avxy->ta',dm2_cas,eri)

    return g_orb - g_orb.T # this gradient is a matrix, be careful we need to pack it before joining it with the CI part

def get_gradCI(self,ci_mat,h1e,eri):
    g_ci = np.zeros(len(ci_mat)-1)
    ciO = ci_mat[:,0]

    for i in range(len(ci_mat)-1):
        t_dm1_cas, t_dm2_cas = self.get_tCASRDM12(self,ci_mat[:,i+1],ciO)
        t_dm1 = CASRDM1_to_RDM1(t_dm1_cas,self.norb,self.ncore,self.ncas)
        t_dm2 = CASRDM2_to_RDM2(t_dm1_cas,t_dm2_cas,self.norb,self.ncore,self.ncas)

        g_ci[i] = 2*np.einsum('pq,pq',h1e,t_dm1) + np.einsum('pqrs,pqrs',eri,t_dm2)

    return g_ci

def form_Grad(self,g_orb,g_ci):
    uniq_g_orb = self.pack_uniq_var(g_orb)
    g = np.concatenate((g_orb,g_ci))
    return g

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
            self.nelecas = (nelecas[0],nelecas[1])
        self.nDeta = scipy.special.comb(self.ncas,self.nelecas[0])
        self.nDetb = scipy.special.comb(self.ncas,self.nelecas[1])
        self.nDet = self.nDeta*self.nDetb

        self.v1e = mol.intor('int1e_nuc') #Nuclear repulsion matrix elements
        self.t1e = mol.intor('int1e_kin') #Kinetic energy matrix elements
        self.h1e_AO =  self.t1e + self.v1e
        self.eri_AO = mol.intor('int2e') #eri in the AO basis in the chemist notation
        self.frozen = frozen

        self.fcisolver = fci.direct_spin0_symm.FCISolver(mol)

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
            self.mo_coeff = myhf.mo_coeff
            self.h1e = np.einsum('ip,ij,jq->pq', self.mo_coeff, self.h1e_ao, self.mo_coeff) # We transform the 1-electron integrals to the MO basis
            self.eri = np.asarray(mol.ao2mo(self.mo_coeff)) # eri in the MO basis as super index matrix (ij|kl) with i>j and k>l VERIFY THIS LAST POINT
            self.eri = ao2mo.restore(1, self.eri, norb) # eri in the MO basis with chemist notation
            return myhf.mo_coeff
        else:
            self.mo_coeff = initMO
            self.h1e = np.einsum('ip,ij,jq->pq', self.mo_coeff, self.h1e_ao, self.mo_coeff) # We transform the 1-electron integrals to the MO basis
            self.eri = np.asarray(mol.ao2mo(self.mo_coeff)) # eri in the MO basis as super index matrix (ij|kl) with i>j and k>l VERIFY THIS LAST POINT
            self.eri = ao2mo.restore(1, self.eri, norb) # eri in the MO basis with chemist notation
            return self._initMO

    @property #TODO add an option to initialize as a full CASCI diagonalization instead of a diagonal of 1
    def initCI(self):
        if self._initCI is None:
            self.CI_mat = np.diag(np.ones(nDet))
            return np.diag(np.ones(nDet))
        else:
            self.CI_mat = initCI
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
        return get_CASRDM_1(self,ci,self.ncas,self.nelecas)

    def get_CASRDM_12(self,ci):
        ''' This method compute the 1-RDM and 2-RDM in the CAS space of a given ci wave function '''
        return get_CASRDM_12(self,ci,self.ncas,self.nelecas)

    def CASRDM1_to_RDM1(self,casdm1):
        ''' This method takes a 1-RDM in the CAS space and transform it to the full MO space '''
        return CASRDM1_to_RDM1(casdm1,self.norb,self.ncore,self.ncas)

    def CASRDM2_to_RDM2(self,casdm2):
        ''' This method takes a 2-RDM in the CAS space and transform it to the full MO space '''
        return CASRDM2_to_RDM2(casdm2,self.norb,self.ncore,self.ncas)

    def get_tCASRDM1(self,ci1,ci2):
        ''' This method compute the 1-electron transition density matrix between the ci vectors ci1 and ci2 '''
        return get_tCASRDM1(self,ci1,ci2,self.ncas,self.nelecas)

    def get_tCASRDM12(self,ci1,ci2):
        ''' This method compute the 1- and 2-electrons transition density matrix between the ci vectors ci1 and ci2 '''
        return get_tCASRDM12(self,ci1,ci2,self.ncas,self.nelecas)

    def get_genFock(self,dm1_cas):
        ''' This method build the generalized Fock matrix '''
        return get_F_core(self.norb,self.ncore,self.h1e,self.eri) + get_F_cas(self.norb,self.ncas,dm1_cas,self.eri)

    def uniq_var_indices(self, nmo, frozen):
        ''' This function creates a matrix of boolean of size (norb,norb). A True element means that this rotation should be taken into account during the optimization. Taken from pySCF.mcscf.casscf '''
        nocc = self.ncore + self.ncas
        mask = np.zeros((self.norb,self.norb),dtype=bool)
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
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        return mat[idx]

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        ''' This method transforms a list of unique rotations elements into an anti-symmetric rotation matrix. Taken from pySCF.mcscf.casscf '''
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        mat = np.zeros((nmo,nmo))
        mat[idx] = v
        return mat - mat.T

    def get_GradOrb(self,F_core,F_cas,dm1_cas,dm2_cas): #TODO add an if statement for MCSCF case
        ''' This method build the orbital part of the gradient '''
        return get_GradOrb(self,F_core,F_cas,dm1_cas,dm2_cas)

    def get_GradCI(self,ci_mat):
        ''' This method build the CI part of the gradient '''
        return get_gradCI(self,ci_mat,self.h1e,self.eri)

    def form_Grad(self,g_orb,g_ci):
        ''' This method concatenate the orbital and CI part of the gradient '''
        return form_Grad(self,g_orb,g_ci)

    def get_Hessian(self): #TODO
        ''' This method concatenate the orb-orb, orb-CI and CI-CI part of the Hessian '''
        pass

    def kernel(): #TODO
        ''' This method runs the iterative Newton-Raphson loop '''
        pass

##### Main #####
if __name__ == '__main__':

    mol = pyscf.M(
        atom = 'H 0 0 0; H 0 0 1.2',
        basis = '6-31g')

    myhf = mol.RHF().run()

    mycas = NR_CASSCF(myhf,2,2)


