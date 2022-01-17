#!/usr/bin/env python
# Author: Antoine Marie

import sys
from functools import reduce
import numpy as np
import scipy.linalg
import re
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

def kernel(self):
    print("Initialization of the Newton-Raphson loop")
    self.initMO # We initialize the quantities that need it
    self.initCI

    self.initializeMO()
    self.initializeCI()

    self.initHeff

    mycas.check_sanity() # Check that the definition of the CAS by the user is sane

    enuc = self._scf.energy_nuc()

    step = 0
    conv = 1

    print("Start of the iterative loop \n")

    dm1_cas, dm2_cas = self.get_CASRDM_12(self.mat_CI[:,0])

    dm1 = self.CASRDM1_to_RDM1(dm1_cas)
    dm2 = self.CASRDM2_to_RDM2(dm1_cas,dm2_cas)

    while conv > self.conv_threshold and step < self.max_iterations:

        # Compute gradient and Hessian
        g_orb = self.get_gradOrb(dm1_cas, dm2_cas)
        g_ci = self.get_gradCI()
        g = self.form_grad(g_orb,g_ci)

        nIndepRot = len(g) - len(g_ci)

        H = self.get_hessian()

        # Update rotation parameters
        NR = -1*np.dot(scipy.linalg.pinv(H),g)

        NR_Orb = NR[:nIndepRot]
        NR_Orb = self.unpack_uniq_var(NR_Orb)

        NR_CI = NR[nIndepRot:]
        S = np.zeros((self.nDet,self.nDet))
        for k in range(1,self.nDet):
            for i in range(self.nDet):
                for j in range(self.nDet):
                    S[i,j] += NR_CI[k-1]*(self.mat_CI[i,k]*self.mat_CI[j,0] - self.mat_CI[j,k]*self.mat_CI[i,0])

        self.mo_coeff = self.rotateOrb(NR_Orb)
        self.mat_CI = self.rotateCI(S)

        # print("This is the updated mo_coeff", self.mo_coeff)
        # print("This is the updated mat_CI", self.mat_CI)

        # Update integrals and density matrices
        self.h1e = np.einsum('ip,ij,jq->pq', self.mo_coeff, self.h1e_AO, self.mo_coeff)
        self.eri = np.asarray(mol.ao2mo(self.mo_coeff))
        self.eri = ao2mo.restore(1, self.eri, self.norb)
        self.h1eff, self.energy_core = self.h1e_for_cas(self.mo_coeff)
        self.h2eff = self.get_h2eff(self.mo_coeff)
        self.h2eff = ao2mo.restore(1,self.h2eff,self.ncas)
        dm1_cas, dm2_cas = self.get_CASRDM_12(self.mat_CI[:,0])
        dm1 = self.CASRDM1_to_RDM1(dm1_cas)
        dm2 = self.CASRDM2_to_RDM2(dm1_cas,dm2_cas)

        # print("This is the updated dm1_cas", dm1_cas)
        # print("This is the updated dm2_cas", dm2_cas)

        # nrj = self.get_energy(self.h1e, self.eri, dm1_cas, dm2_cas)
        # print("This is the updated nrj", nrj + enuc)
        # nrj = self.get_energy_cas(self.h1eff, self.h2eff, dm1_cas, dm2_cas)
        # # print("This is the updated nrj", nrj + enuc + self.energy_core)

        conv = np.max(np.abs(g))
        # print("At this iteration the convergence is ", conv)

        step += 1

    self.conv = conv
    self.nb_it = step

    if step==self.max_iterations:
        print("The Newton-Raphson has not converged!!!")
        return

    else:
        print("The Newton-Raphson has converged in ", step, " steps.\n")

    if self.ncore==0:
        nrj = self.get_energy_cas(self.h1eff, self.h2eff, dm1_cas, dm2_cas)
        print("The energy at convergence is ", nrj + enuc, "\n")
        self.e_tot = nrj + enuc
    else:
        nrj = self.get_energy_cas(self.h1eff, self.h2eff, dm1_cas, dm2_cas)
        print("This energy at convergence is ", nrj + enuc + self.energy_core, "\n")
        self.e_tot = nrj + enuc + self.energy_core

    print("This is the MO coefficients at convergence\n")
    matprint(self.mo_coeff)
    print("")
    print("This is the CI coefficients at convergence\n")
    matprint(self.mat_CI)
    print("")
    print("This is the CAS DM1 at convergence\n")
    matprint(dm1_cas)
    print("")
    # print("This is the trace of the 1 cas dm", np.trace(dm1_cas))

    spin, mul = self.spin_square(self.mat_CI[:,0])
    print("The squared spin value of the wave function is ", spin, " and its associated multiplicity is ", mul, ".\n")
    return

def grid_point(nb_point, n, length):
    ''' This function compute the decomposition of a number n in the nb_point basis. For example if nb_point=2, it gives the binary decomposition. This is used to build the grid. '''
    if n == 0:
        return np.zeros((length))
    nums = []
    while n:
        n, r = divmod(n, nb_point)
        nums.append(r)
    while len(nums)<length:
        nums.append(0)
    nums.reverse()
    return nums

def grid_search(self):
    ''' This function runs the Newton-Raphson algorithm for a grid of starting point '''
    self.initMO # We initialize the quantities that need it
    self.initCI
    self.initializeMO()
    self.initializeCI()
    self.initHeff

    mycas.check_sanity() # Check that the definition of the CAS by the user is sane

    print("Start of the grid calculation\n")

    print("The initial MOs are\n")
    matprint(self.mo_coeff)
    print("")
    print("The initial CI coefficients are\n")
    matprint(self.mat_CI)
    print("")

    nb_point = 3
    Nb_CI_point = nb_point**(self.nDet - 1) # Number of CI points on the grid
    Nb_orb_point = int(nb_point**(0.5*self.norb*(self.norb-1))) # Number of orbitals points on the grid

    print("There are ", nb_point, " points per rotation elements")
    print("This gives ", Nb_CI_point, " CI points and ",Nb_orb_point, " orbital points.\n")

    iterator = 0

    # Grid loop
    for orb_gridpoint in range(Nb_orb_point):
        index_orb = grid_point(nb_point, orb_gridpoint, int(0.5*self.norb*(self.norb-1)))
        K = np.zeros((self.norb, self.norb))
        K[np.triu_indices(self.norb, 1)] = index_orb
        K = np.asarray(K - K.T)*0.5*np.pi # Create the rotation associated to index_orb
        K = self.rotateOrb(K) # Rotate the mo coeff

        for ci_gridpoint in range(Nb_CI_point):
            index_ci = grid_point(nb_point, ci_gridpoint, (self.nDet - 1))
            S = np.zeros((self.nDet, self.nDet))
            S[0,1:] = index_ci
            S = np.asarray(S - S.T)*(1/nb_point)*np.pi #Create the rotation associated to index_ci
            S = self.rotateCI(S) # Rotate the ci coeff
            iterator += 1

            # Run the calculations
            print("Start the Newton-Raphson calcualtion number", iterator, ".\n")
            print("The mo coefficients are rotated by ", index_orb, " and the CI coefficients are rotated by ", index_ci, ".\n")
            tmp_cas = NR_CASSCF(self._scf,self.ncas,self.nelecas,ncore=self.ncore,initMO=K,initCI=S,frozen=self.frozen)
            tmp_cas.kernel()

            index_neg, index_pos, nb_zero = tmp_cas.get_index()
            print("The hessian of this solution has ", index_neg," negative eigenvalues, ", index_pos, " positive eigenvalues and ", nb_zero," zero eigenvalues.\n")
    return

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

    def __init__(self,myhf_or_mol,ncas,nelecas,ncore=None,initMO = None, initCI = None,frozen=None):
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
            self.nelecas = np.asarray((nelecas[0],nelecas[1])).astype(int)
        self.v1e = mol.intor('int1e_nuc')       # Nuclear repulsion matrix elements
        self.t1e = mol.intor('int1e_kin')       # Kinetic energy matrix elements
        self.h1e_AO =  self.t1e + self.v1e      # 1-electron matrix elements in the AO basis
        self.norb = len(self.h1e_AO)            # Number of orbitals
        self.nDeta = (scipy.special.comb(self.ncas,self.nelecas[0])).astype(int)
        self.nDetb = (scipy.special.comb(self.ncas,self.nelecas[1])).astype(int)
        self.nDet = (self.nDeta*self.nDetb).astype(int)
        self.eri_AO = mol.intor('int2e')        # ERI in the AO basis in the chemist notation
        self._ncore = ncore                     # Number of core orbitals
        self.frozen = frozen                    # Number of frozen orbitals

        self._initMO = initMO
        self._initCI = initCI

        self.mo_coeff = None
        self.mat_CI = None

        self.h1eff = None
        self.h2eff = None

        self.fcisolver = fci.direct_spin1.FCISolver(mol)

        self.conv_threshold = 1e-08
        self.max_iterations = 512

        self.e_tot = None
        self.conv = None
        self.nb_it = None

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
        if self._initMO is None:
            self._initMO = self._scf.mo_coeff
            return
        else :
            return self._initMO

    @property
    def initCI(self):
        if self._initCI is None:
            self._initCI = np.identity(self.nDet, dtype="float")
            return
        else:
            return self._initCI

    # @property
    # def mo_coeff(self):
    #     return self.mo_coeff
    #
    # @mo_coeff.setter
    # def mo_coeff(self,value):
    #     self.mo_coeff = value


    # @mo_coeff.setter
    # def mo_coeff(self):
    #     if self.mo_coeff is None:
    #         self.mo_coeff = self._scf.mo_coeff
    #         self.h1e = np.einsum('ip,ij,jq->pq', self._scf.mo_coeff, self.h1e_AO, self._scf.mo_coeff) # We transform the 1-electron integrals to the MO basis
    #         self.eri = np.asarray(mol.ao2mo(self._scf.mo_coeff)) # eri in the MO basis as super index matrix (ij|kl) with i>j and k>l VERIFY THIS LAST POINT
    #         self.eri = ao2mo.restore(1, self.eri, self.norb) # eri in the MO basis with chemist notation

    # @property
    # def mat_CI(self, value):
    #     if self.mat_CI is None:
    #         self.mat_CI = np.identity(self.nDet, dtype="float")

    def initializeMO(self):
        self.mo_coeff = self._initMO
        self.h1e = np.einsum('ip,ij,jq->pq', self._initMO, self.h1e_AO, self._initMO) # We transform the 1-electron integrals to the MO basis
        self.eri = np.asarray(mol.ao2mo(self._initMO)) # eri in the MO basis as super index matrix (ij|kl) with i>j and k>l VERIFY THIS LAST POINT
        self.eri = ao2mo.restore(1, self.eri, self.norb) # eri in the MO basis with chemist notation

    def initializeCI(self):
        self.mat_CI = self._initCI


    @property
    def initHeff(self):
        self.h1eff, self.energy_core = self.h1e_for_cas()
        self.h2eff = self.get_h2eff()
        self.h2eff = ao2mo.restore(1,self.h2eff,self.ncas)
        return

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

    #Taken from pyscf
    def get_jk(self, mol, dm, hermi=1, with_j=True, with_k=True, omega=None):
        return self._scf.get_jk(mol, dm, hermi,
                                with_j=with_j, with_k=with_k, omega=omega)

    def get_veff(self, mol=None, dm=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None:
            mocore = self.mo_coeff[:,:self.ncore]
            dm = np.dot(mocore, mocore.conj().T) * 2
        # don't call self._scf.get_veff because _scf might be DFT object
        vj, vk = self.get_jk(mol, dm, hermi)
        return vj - vk * .5

    def h1e_for_cas(self, mo_coeff=None, ncas=None, ncore=None):
        '''CAS sapce one-electron hamiltonian

        Args:
            casci : a CASSCF/CASCI object or RHF object

        Returns:
            A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
            the second is the electronic energy from core.
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ncas is None: ncas = self.ncas
        if ncore is None: ncore = self.ncore
        mo_core = mo_coeff[:,:ncore]
        mo_cas = mo_coeff[:,ncore:ncore+ncas]

        hcore = self.get_hcore()
        energy_core = self.energy_nuc()
        if mo_core.size == 0:
            corevhf = 0
        else:
            core_dm = np.dot(mo_core, mo_core.conj().T) * 2
            corevhf = self.get_veff(self.mol, core_dm)
            energy_core += np.einsum('ij,ji', core_dm, hcore).real
            energy_core += np.einsum('ij,ji', core_dm, corevhf).real * .5
        h1eff = reduce(np.dot, (mo_cas.conj().T, hcore+corevhf, mo_cas))
        return h1eff, energy_core

    def get_h2eff(self, mo_coeff=None):
        '''Compute the active space two-particle Hamiltonian. '''
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        if mo_coeff is None:
            ncore = self.ncore
            mo_coeff = self.mo_coeff[:,ncore:nocc]
        elif mo_coeff.shape[1] != ncas:
            mo_coeff = mo_coeff[:,ncore:nocc]

        if self._scf._eri is not None:
            eri = ao2mo.full(self._scf._eri, mo_coeff,
                             max_memory=self.max_memory)
        else:
            eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
                             max_memory=self.max_memory)
        return eri


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

        return dm1_cas.T, 0.5*dm2_cas # We transpose the 1-RDM because their convention is <|a_q^\dagger a_p|>

    def get_tCASRDM1(self,ci1,ci2):
        ''' This method computes the 1-electron transition density matrix between the ci vectors ci1 and ci2 '''
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

        return t_dm1_cas.T

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

        #t_dm2_cas = t_dm2_cas + np.einsum('pqrs->qpsr',t_dm2_cas) #TODO check this

        return t_dm1_cas.T, 0.5*t_dm2_cas

    def CASRDM1_to_RDM1(self,dm1_cas, transition=False):
        ''' This method takes a 1-RDM in the CAS space and transform it to the full MO space '''
        ncore = self.ncore
        ncas = self.ncas
        dm1 = np.zeros((self.norb,self.norb))
        if transition is False:
            if ncore > 0:
                dm1_core = 2*np.identity(self.ncore, dtype="int") #the OccOcc part of dm1 is a diagonal of 2
                dm1[:ncore,:ncore] = dm1_core
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
            dm1 = mycas.CASRDM1_to_RDM1(dm1_cas)
            for i in range(ncore):
                for j in range(ncore):
                    for p in range(ncore,ncore+ncas):
                        for q in range(ncore,ncore+ncas):
                            dm2[i,j,p,q] = delta_kron(i,j)*dm1[p,q] - delta_kron(i,q)*delta_kron(j,p)
                            dm2[p,q,i,j] = dm2[i,j,p,q]

                            dm2[p,i,j,q] = 2*delta_kron(i,p)*delta_kron(j,q) - 0.5*delta_kron(i,j)*dm1[p,q]
                            dm2[j,q,p,i] = dm2[p,i,j,q]

            for i in range(ncore):
                for j in range(ncore):
                    for k in range(ncore):
                        for l in range(ncore):
                            dm2[i,j,k,l] = 2*delta_kron(i,j)*delta_kron(k,l) - delta_kron(i,l)*delta_kron(j,k)

        if transition is True:
            dm1 = mycas.CASRDM1_to_RDM1(dm1_cas,True)
            for i in range(ncore):
                for j in range(ncore):
                    for p in range(ncore,ncore+ncas):
                        for q in range(ncore,ncore+ncas):
                            dm2[i,j,p,q] = delta_kron(i,j)*dm1[p,q]
                            dm2[p,q,i,j] = delta_kron(i,j)*dm1[q,p]

                            dm2[p,i,j,q] = -0.5*delta_kron(i,j)*dm1[p,q]
                            dm2[j,q,p,i] = -0.5*delta_kron(i,j)*dm1[q,p]

        # dm2 = 0.5*(dm2 + np.einsum("pqrs->qprs",dm2)) #According to QC and dynamics of excited states not sure about this

        dm2[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc] = dm2_cas # Finally we add the uvxy sector
        return dm2

    @staticmethod
    def check_symmetry(tensor):
        if np.allclose(tensor,np.einsum('pqrs->rspq',tensor),atol=1e-06):
            print("The tensor have the PQRS/RSPQ symmetry")
        if np.allclose(tensor,np.einsum('pqrs->qpsr',tensor),atol=1e-06):
            print("The tensor have the PQRS/QPSR symmetry")
        if np.allclose(tensor,np.einsum('pqrs->qprs',tensor),atol=1e-06):
            print("The tensor have the PQRS/QPRS symmetry")
        return

    def get_hamiltonian(self):
        ''' This method build the Hamiltonian matrix '''
        ncore = self.ncore
        ncas = self.ncas
        norb = self.norb
        nocc = ncore + ncas
        nvir = norb - nocc

        H = np.zeros((self.nDet,self.nDet))

        h1e = self.h1e
        eri = self.eri

        id = np.identity(self.nDet)
        for i in range(self.nDet):
            for j in range(self.nDet):
                dm1_cas, dm2_cas = self.get_tCASRDM12(id[:,i],id[:,j])
                dm1 = self.CASRDM1_to_RDM1(dm1_cas,True)
                dm2 = self.CASRDM2_to_RDM2(dm1_cas,dm2_cas,True)
                H[i,j] = np.einsum('pq,pq',h1e,dm1) + np.einsum('pqrs,pqrs',eri,dm2)
        return H

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
        idx = self.uniq_var_indices(self.norb,self.frozen)
        return mat[idx]

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        ''' This method transforms a list of unique rotations elements into an anti-symmetric rotation matrix. Taken from pySCF.mcscf.casscf '''
        idx = self.uniq_var_indices(self.norb, self.frozen)
        mat = np.zeros((self.norb,self.norb))
        mat[idx] = v
        return mat - mat.T

    def get_F_core(self):
        ncore = self.ncore
        return(self.h1e + 2*np.einsum('pqii->pq', self.eri[:, :, :ncore, :ncore]) - np.einsum('piiq->pq', self.eri[:, :ncore, :ncore, :]))

    def get_F_cas(self,dm1_cas):
        ncore = self.ncore
        nocc = ncore + self.ncas
        return(np.einsum('tu,pqtu->pq', dm1_cas, self.eri[:, :, ncore:nocc, ncore:nocc]) - 0.5*np.einsum('tu,puqt->pq', dm1_cas, self.eri[:, ncore:nocc, :, ncore:nocc]))

    def get_gradOrb(self,dm1_cas,dm2_cas):
        ''' This method builds the orbital part of the gradient '''
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
            g_orb[ncore:nocc,:ncore] = 4*(F_core + F_cas)[ncore:nocc,:ncore] - 2*np.einsum('tv,iv->ti', dm1_cas, F_core[:ncore,ncore:nocc]) - 4*np.einsum('tvxy,ivxy->ti', dm2_cas, self.eri[:ncore,ncore:nocc,ncore:nocc,ncore:nocc])
        #virtual-active rotations g_{at}
        if nvir>0:
            g_orb[nocc:,ncore:nocc] = 2*np.einsum('tv,av->at', dm1_cas, F_core[nocc:,ncore:nocc]) + 4*np.einsum('tvxy,avxy->at',dm2_cas, self.eri[nocc:,ncore:nocc,ncore:nocc,ncore:nocc])

        return g_orb - g_orb.T # this gradient is a matrix, be careful we need to pack it before joining it with the CI part

    # def get_gradOrbOK(self,dm1_cas,dm2_cas):
    #     g_orb = np.zeros((self.norb,self.norb))
    #     ncore = self.ncore
    #     ncas = self.ncas
    #     nocc = ncore + ncas
    #     nvir = self.norb - nocc
    #     dm1 = self.CASRDM1_to_RDM1(dm1_cas)
    #     dm2 = self.CASRDM2_to_RDM2(dm1_cas,dm2_cas)
    #     F = np.einsum('xq,qy->xy', dm1, self.h1e) + 2*np.einsum('xqrs,yqrs->xy', dm2, self.eri)
    #     return -2*(F - F.T)

    def get_gradCI(self):
        ''' This method build the CI part of the gradient '''
        mat_CI = self.mat_CI
        g_CI = np.zeros(len(mat_CI)-1)
        ciO = mat_CI[:,0]

        H_fci = self.fcisolver.pspace(self.h1eff, self.h2eff, self.ncas, self.nelecas, np=1000000)[1]

        for k in range(len(mat_CI)-1):

            for i in range(len(mat_CI)):
                for j in range(len(mat_CI)):

                    g_CI[k] += 2*mat_CI[i,0]*H_fci[i,j]*mat_CI[j,k+1]
        return g_CI

    def form_grad(self,g_orb,g_ci):
        ''' This method concatenate the orbital and CI part of the gradient '''
        uniq_g_orb = self.pack_uniq_var(g_orb) #We apply the mask to obtain a list of unique rotations
        g = np.concatenate((uniq_g_orb,g_ci))
        return g

    def get_hessianOrbOrb(self,dm1_cas,dm2_cas):
        ''' This method build the orb-orb part of the hessian '''
        norb = self.norb
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        nvir = norb - nocc

        #H = np.zeros((nrot,nrot))
        H = np.zeros((norb,norb,norb,norb))

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

            H[nocc:, :ncore, nocc:, :ncore] = 4*(4*aibj - abij - ajbi) + 4*np.einsum('ij,ab->aibj', np.identity(ncore), F_tot[nocc:,nocc:]) - 4*np.einsum('ab,ij->aibj', np.identity(nvir), F_tot[:ncore, :ncore])

        #virtual-core virtual-active H_{ai,bt}
        if ncore>0 and nvir>0:
            aibv = eri[nocc:, :ncore, nocc:, ncore:nocc]
            avbi = eri[nocc:, ncore:nocc, nocc:, :ncore]
            abvi = eri[nocc:, nocc:, ncore:nocc, :ncore]

            avbi = np.einsum('avbi->aibv', avbi)
            abvi = np.einsum('abvi->aibv', abvi)

            H[nocc:, :ncore, nocc:, ncore:nocc] = 2*np.einsum('tv,aibv->aibt', dm1_cas, 4*aibv - avbi - abvi) - 2*np.einsum('ab,tvxy,vixy ->aibt', np.identity(nvir), dm2_cas,eri[ncore:nocc, :ncore, ncore:nocc, ncore:nocc]) - 2*np.einsum('ab,ti->aibt', np.identity(nvir), F_tot[ncore:nocc, :ncore]) - np.einsum('ab,tv,vi->aibt', np.identity(nvir), dm1_cas, F_core[ncore:nocc, :ncore])

        #virtual-active virtual-core H_{bt,ai}
        if ncore>0 and nvir>0:
             H[nocc:, ncore:nocc, nocc:, :ncore] = np.einsum('aibt->btai',H[nocc:, :ncore, nocc:, ncore:nocc])

        #virtual-core active-core H_{ai,tj}
        if ncore>0 and nvir>0:
            aivj = eri[nocc:, :ncore, ncore:nocc, :ncore]
            avji = eri[nocc:, ncore:nocc, :ncore, :ncore]
            ajvi = eri[nocc:, :ncore, ncore:nocc, :ncore]

            avji = np.einsum('avji->aivj', avji)
            ajvi = np.einsum('ajvi->aivj', ajvi)

            H[nocc:, :ncore, ncore:nocc, :ncore] = 2*np.einsum('tv,aivj->aitj', (2*np.identity(ncas) - dm1_cas), 4*aivj - avji - ajvi) - 2*np.einsum('ji,tvxy,avxy -> aitj', np.identity(ncore), dm2_cas, eri[nocc:, ncore:nocc, ncore:nocc, ncore:nocc]) + 4*np.einsum('ij,at-> aitj', np.identity(ncore), F_tot[nocc:, ncore:nocc]) - np.einsum('ij,tv,av-> aitj', np.identity(ncore), dm1_cas, F_core[nocc:, ncore:nocc])

        #active-core virtual-core H_{tj,ai}
        if ncore>0 and nvir>0:
            H[ncore:nocc, :ncore, nocc:, :ncore] = np.einsum('aitj->tjai',H[nocc:, :ncore, ncore:nocc, :ncore])

        #virtual-active virtual-active H_{at,bu}
        if nvir>0:
            tmp1 = 4*np.einsum('tuvx,abvx->atbu', dm2_cas, eri[nocc:, nocc:, ncore:nocc, ncore:nocc]) + 4*np.einsum('txvu,axbv->atbu', dm2_cas, eri[nocc:, ncore:nocc, nocc:, ncore:nocc]) + 4*np.einsum('txuv,axbv->atbu', dm2_cas, eri[nocc:, ncore:nocc, nocc:, ncore:nocc])

            tmp2 = - 2*np.einsum('ab,tvxy,uvxy->atbu', np.identity(nvir), dm2_cas, eri[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc]) - np.einsum('ab,tv,uv->atbu',np.identity(nvir), dm1_cas, F_core[ncore:nocc, ncore:nocc])

            tmp3 = - 2*np.einsum('ab,uvxy,tvxy->atbu', np.identity(nvir), dm2_cas, eri[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc]) - np.einsum('ab,uv,tv->atbu', np.identity(nvir), dm1_cas, F_core[ncore:nocc, ncore:nocc])

            H[nocc:, ncore:nocc, nocc:, ncore:nocc] = tmp1 + tmp2 + tmp3 + 2*np.einsum('tu,ab->atbu', dm1_cas, F_core[nocc:, nocc:])

            # H[ncore:nocc, nocc:, ncore:nocc, nocc:] = np.einsum('atbu->taub', H[nocc:, ncore:nocc, nocc:, ncore:nocc])
            # H[ncore:nocc, nocc:, nocc:, ncore:nocc] = np.einsum('atbu->tabu', H[nocc:, ncore:nocc, nocc:, ncore:nocc])

        #active-core virtual-active H_{ti,au}
        if ncore>0 and nvir>0:
            tmp1 = - 4*np.einsum('tuvx,aivx->tiau', dm2_cas, eri[nocc:, :ncore, ncore:nocc, ncore:nocc]) - 4*np.einsum('tvux,axvi->tiau', dm2_cas, eri[nocc:, ncore:nocc, ncore:nocc, :ncore]) - 4*np.einsum('tvxu,axvi->tiau', dm2_cas, eri[nocc:, ncore:nocc, ncore:nocc, :ncore])

            avti = eri[nocc:, ncore:nocc, ncore:nocc, :ncore]
            aitv = eri[nocc:, :ncore, ncore:nocc, ncore:nocc]
            atvi = eri[nocc:, ncore:nocc, ncore:nocc, :ncore]

            aitv = np.einsum('aitv->avti', aitv)
            atvi = np.einsum('atvi->avti', atvi)

            tmp2 = 2*np.einsum('uv,avti->tiau', dm1_cas, 4*avti - aitv - atvi) - 2*np.einsum('tu,ai->tiau', dm1_cas, F_core[nocc:, :ncore]) + 2*np.einsum('tu,ai->tiau',np.identity(ncas),F_tot[nocc:, :ncore])

            H[ncore:nocc, :ncore, nocc:, ncore:nocc] = tmp1 + tmp2

        #virtual-active active-core  H_{au,ti}
            H[nocc:, ncore:nocc, ncore:nocc, :ncore]  = np.einsum('auti->tiau',H[ncore:nocc, :ncore, nocc:, ncore:nocc] )

        #active-core active-core H_{ti,uj}
        if ncore>0:
            tmp1 = 4*np.einsum('utvx,vxij->tiuj', dm2_cas, eri[ncore:nocc, ncore:nocc, :ncore, :ncore]) + 4*np.einsum('uxvt,vixj->tiuj', dm2_cas, eri[ncore:nocc, :ncore, ncore:nocc, :ncore]) + 4*np.einsum('uxtv,vixj->tiuj', dm2_cas, eri[ncore:nocc, :ncore, ncore:nocc, :ncore])

            viuj = eri[ncore:nocc, :ncore, ncore:nocc, :ncore]
            uivj = eri[ncore:nocc, :ncore, ncore:nocc, :ncore]
            uvij = eri[ncore:nocc, ncore:nocc, :ncore, :ncore]

            uivj = np.einsum('uivj->viuj', uivj)
            uvij = np.einsum('uvij->viuj', uvij)

            tmp2 = 2*np.einsum('tv,viuj->tiuj', np.identity(ncas) - dm1_cas, 4*viuj - uivj - uvij)

            tmp2 = tmp2 + np.einsum('tiuj->uitj', tmp2)

            tmp3 = 2*np.einsum('tu,ij->tiuj', dm1_cas, F_core[:ncore, :ncore]) - 4*np.einsum('ij,tvxy,uvxy->tiuj', np.identity(ncore), dm2_cas, eri[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc]) - 2*np.einsum('ij,uv,tv->tiuj', np.identity(ncore), dm1_cas, F_core[ncore:nocc, ncore:nocc])

            tmp4 = 4*np.einsum('ij,tu->tiuj', np.identity(ncore), F_tot[ncore:nocc, ncore:nocc]) - 4*np.einsum('tu,ij->tiuj', np.identity(ncas), F_tot[:ncore, :ncore])

            H[ncore:nocc, :ncore, ncore:nocc, :ncore] = tmp1 + tmp2 + tmp3 + tmp4

            H[ncore:nocc, :ncore, ncore:nocc, :ncore] = 0.5*(H[ncore:nocc, :ncore, ncore:nocc, :ncore] + np.einsum('tiuj->ujti',H[ncore:nocc, :ncore, ncore:nocc, :ncore])) #I have to think about this ...

        return(H)

    def get_hessianCICI(self):
        ''' This method build the CI-CI part of the hessian '''
        mat_CI = self.mat_CI
        hessian_CICI = np.zeros((len(mat_CI)-1,len(mat_CI)-1))

        H_fci = self.fcisolver.pspace(self.h1eff, self.h2eff, self.ncas, self.nelecas, np=1000000)[1]

        c0 = mat_CI[:,0]
        e0 = np.einsum('i,ij,j',c0, H_fci, c0)

        for k in range(1,len(mat_CI)): # Loop on Hessian indices
                cleft = mat_CI[:,k]
                for l in range(1,len(mat_CI)):
                    cright = mat_CI[:,l]
                    hessian_CICI[k-1,l-1] = 2*np.einsum('i,ij,j',cleft, H_fci, cright) - 2*delta_kron(k,l)*e0
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
                dm1 = self.CASRDM1_to_RDM1(dm1_cas,True)
                dm2 = self.CASRDM2_to_RDM2(dm1_cas,dm2_cas,True)
                if ncore>0 and nvir>0:
                    one_el = np.einsum('pa,pi->ai', h1e[:, nocc:], dm1[:, :ncore]) + np.einsum('ap,ip->ai', h1e[nocc:, :], dm1[:ncore, :])

                    two_el = np.einsum('pars,pirs->ai', eri[:, nocc:, :, :], dm2[:, :ncore, :, :]) + np.einsum('aqrs,iqrs->ai', eri[nocc:, :, :, :], dm2[:ncore, :, :, :]) + np.einsum('pqra,pqri->ai', eri[:, :, :, nocc:], dm2[:, :, :, :ncore]) + np.einsum('pqas,pqis->ai', eri[:, :, nocc:, :], dm2[:, :, :ncore, :])

                    H_ai[:, :, i, j] = one_el + two_el
                if nvir>0:
                    one_el = np.einsum('pa,pt->at', h1e[:, nocc:], dm1[:, ncore:nocc]) + np.einsum('ap,tp->at', h1e[nocc:, :], dm1[ncore:nocc, :])

                    two_el = np.einsum('pars,ptrs->at', eri[:, nocc:, :, :], dm2[:, ncore:nocc, :, :]) + np.einsum('aqrs,tqrs->at', eri[nocc:, :, :, :], dm2[ncore:nocc, :, :, :]) + np.einsum('pqra,pqrt->at', eri[:, :, :, nocc:], dm2[:, :, :, ncore:nocc]) + np.einsum('pqas,pqts->at', eri[:, :, nocc:, :], dm2[:, :, ncore:nocc, :])

                    H_at[:, :, i, j] = one_el + two_el

                if ncore>0:
                    one_el = np.einsum('pt,pi->ti', h1e[:, ncore:nocc], dm1[:, :ncore]) - np.einsum('pi,pt->ti', h1e[:, :ncore], dm1[:, ncore:nocc]) + np.einsum('tp,ip->ti', h1e[ncore:nocc, :], dm1[:ncore, :]) - np.einsum('ip,tp->ti', h1e[:ncore, :], dm1[ncore:nocc, :])
                    two_el = np.einsum('ptrs,pirs->ti', eri[:, ncore:nocc, :, :], dm2[:, :ncore, :, :]) - np.einsum('pirs,ptrs->ti', eri[:, :ncore, :, :], dm2[:, ncore:nocc, :, :]) + np.einsum('tqrs,iqrs->ti', eri[ncore:nocc, :, :, :], dm2[:ncore, :, :, :]) - np.einsum('iqrs,tqrs->ti', eri[:ncore, :, :, :], dm2[ncore:nocc, :, :, :]) + np.einsum('pqrt,pqri->ti', eri[:, :, :, ncore:nocc], dm2[:, :, :, :ncore]) - np.einsum('pqri,pqrt->ti', eri[:, :, :, :ncore], dm2[:, :, :, ncore:nocc]) + np.einsum('pqts,pqis->ti', eri[:, :, ncore:nocc, :], dm2[:, :, :ncore, :]) - np.einsum('pqis,pqts->ti', eri[:, :, :ncore, :], dm2[:, :, ncore:nocc, :])
                    H_ti[:, :, i, j] = one_el + two_el

        return H_ai, H_at, H_ti

    def get_hessianOrbCI(self): #TODO
        ''' This method build the orb-CI part of the hessian '''
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        nvir = self.norb - nocc
        H_OCI = np.zeros((self.norb,self.norb,self.nDet-1))
        mat_CI = self.mat_CI

        H_ai, H_at, H_ti = self.get_hamiltonianComm()

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

    def get_hessian(self): #TODO
        ''' This method concatenate the orb-orb, orb-CI and CI-CI part of the Hessian '''
        norb = self.norb
        nDet = self.nDet

        idx = self.uniq_var_indices(norb,self.frozen)

        dm1_cas, dm2_cas = self.get_CASRDM_12(self.mat_CI[:,0])

        H_OrbOrb = self.get_hessianOrbOrb(dm1_cas,dm2_cas)
        H_CICI = self.get_hessianCICI()
        H_OrbCI = self.get_hessianOrbCI()

        H_OrbCI = H_OrbCI[idx,:]
        H_OrbOrb = H_OrbOrb[:,:,idx]
        H_OrbOrb = H_OrbOrb[idx,:]

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

    def rotateCI(self,S):
        ci = np.dot(scipy.linalg.expm(S),self.mat_CI)
        return ci

    def numericalGrad(self):
        epsilon = 0.0000001
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
        e0_cas = self.get_energy_cas(self.h1eff, self.h2eff, dm1_cas, dm2_cas)
        for k in range(1,self.nDet):
            S = np.zeros((self.nDet,self.nDet))
            Sk0 = epsilon
            for i in range(self.nDet):
                for j in range(self.nDet):
                    S[i,j] += Sk0*(self.mat_CI[i,k]*self.mat_CI[j,0] - self.mat_CI[j,k]*self.mat_CI[i,0])
            ciUpdate = self.rotateCI(S)
            dm1_casUpdate, dm2_casUpdate = self.get_CASRDM_12(ciUpdate[:,0])
            eUpdate = self.get_energy_cas(self.h1eff, self.h2eff, dm1_casUpdate, dm2_casUpdate)
            g_CI[k-1] = (eUpdate - e0_cas)/epsilon

        g = np.concatenate((g_orb,g_CI))

        return g

    def numericalHessian(self):
        epsilon = 0.0001
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
                        Kpp = np.zeros((norb,norb))
                        Kpp[p,q] += epsilon
                        Kpp[q,p] += -epsilon
                        Kpp[r,s] += epsilon
                        Kpp[s,r] += -epsilon

                        Kpm = np.zeros((norb,norb))
                        Kpm[p,q] += epsilon
                        Kpm[q,p] += -epsilon
                        Kpm[r,s] += -epsilon
                        Kpm[s,r] += epsilon

                        Kmp = np.zeros((norb,norb))
                        Kmp[p,q] += -epsilon
                        Kmp[q,p] += epsilon
                        Kmp[r,s] += epsilon
                        Kmp[s,r] += -epsilon

                        Kmm = np.zeros((norb,norb))
                        Kmm[p,q] += -epsilon
                        Kmm[q,p] += epsilon
                        Kmm[r,s] += -epsilon
                        Kmm[s,r] += epsilon

                        mo_coeffpp = self.rotateOrb(Kpp)
                        mo_coeffpm = self.rotateOrb(Kpm)
                        mo_coeffmp = self.rotateOrb(Kmp)
                        mo_coeffmm = self.rotateOrb(Kmm)

                        h1eUpdatepp = np.einsum('ip,ij,jq->pq', mo_coeffpp, self.h1e_AO, mo_coeffpp)
                        h1eUpdatepm = np.einsum('ip,ij,jq->pq', mo_coeffpm, self.h1e_AO, mo_coeffpm)
                        h1eUpdatemp = np.einsum('ip,ij,jq->pq', mo_coeffmp, self.h1e_AO, mo_coeffmp)
                        h1eUpdatemm = np.einsum('ip,ij,jq->pq', mo_coeffmm, self.h1e_AO, mo_coeffmm)

                        eriUpdatepp = np.asarray(mol.ao2mo(mo_coeffpp))
                        eriUpdatepp = ao2mo.restore(1, eriUpdatepp, norb)
                        eriUpdatepm = np.asarray(mol.ao2mo(mo_coeffpm))
                        eriUpdatepm = ao2mo.restore(1, eriUpdatepm, norb)
                        eriUpdatemp = np.asarray(mol.ao2mo(mo_coeffmp))
                        eriUpdatemp = ao2mo.restore(1, eriUpdatemp, norb)
                        eriUpdatemm = np.asarray(mol.ao2mo(mo_coeffmm))
                        eriUpdatemm = ao2mo.restore(1, eriUpdatemm, norb)

                        eUpdatepp = self.get_energy(h1eUpdatepp, eriUpdatepp, dm1_cas, dm2_cas)
                        eUpdatepm = self.get_energy(h1eUpdatepm, eriUpdatepm, dm1_cas, dm2_cas)
                        eUpdatemp = self.get_energy(h1eUpdatemp, eriUpdatemp, dm1_cas, dm2_cas)
                        eUpdatemm = self.get_energy(h1eUpdatemm, eriUpdatemm, dm1_cas, dm2_cas)

                        H_OrbOrb[p,q,r,s] = (eUpdatepp + eUpdatemm - eUpdatepm - eUpdatemp)/(4*(epsilon**2))

        # print(H_OrbOrb)
        idx = self.uniq_var_indices(self.norb,self.frozen)
        idxidx = np.einsum('pq,rs->pqrs',idx,idx)
        # print(H_OrbOrb[idxidx])
        H_OrbOrb = H_OrbOrb[:,:,idx]
        H_OrbOrb = H_OrbOrb[idx,:]
        matprint(H_OrbOrb)

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

                eUpdatepp = self.get_energy_cas(self.h1eff, self.h2eff, dm1_casUpdatepp, dm2_casUpdatepp)
                eUpdatepm = self.get_energy_cas(self.h1eff, self.h2eff, dm1_casUpdatepm, dm2_casUpdatepm)
                eUpdatemp = self.get_energy_cas(self.h1eff, self.h2eff, dm1_casUpdatemp, dm2_casUpdatemp)
                eUpdatemm = self.get_energy_cas(self.h1eff, self.h2eff, dm1_casUpdatemm, dm2_casUpdatemm)

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

                h1effp, energy_corep = self.h1e_for_cas(mo_coeff = mo_coeffp)
                h2effp = self.get_h2eff(mo_coeff = mo_coeffp)
                h2effp = ao2mo.restore(1,h2effp,self.ncas)
                h1effm, energy_corem = self.h1e_for_cas(mo_coeff = mo_coeffm)
                h2effm = self.get_h2eff(mo_coeff = mo_coeffm)
                h2effm = ao2mo.restore(1,h2effm,self.ncas)

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

                    # eUpdatepp = self.get_energy(h1eUpdatep, eriUpdatep, dm1_casUpdatep, dm2_casUpdatep)
                    # eUpdatepm = self.get_energy(h1eUpdatep, eriUpdatep, dm1_casUpdatem, dm2_casUpdatem)
                    # eUpdatemp = self.get_energy(h1eUpdatem, eriUpdatem, dm1_casUpdatep, dm2_casUpdatep)
                    # eUpdatemm = self.get_energy(h1eUpdatem, eriUpdatem, dm1_casUpdatem, dm2_casUpdatem)
                    eUpdatepp = self.get_energy_cas(h1effp, h2effp, dm1_casUpdatep, dm2_casUpdatep)
                    eUpdatepm = self.get_energy_cas(h1effp, h2effp, dm1_casUpdatem, dm2_casUpdatem)
                    eUpdatemp = self.get_energy_cas(h1effm, h2effm, dm1_casUpdatep, dm2_casUpdatep)
                    eUpdatemm = self.get_energy_cas(h1effm, h2effm, dm1_casUpdatem, dm2_casUpdatem)


                    H_OrbCI[p,q,k-1] = (eUpdatepp + eUpdatemm - eUpdatepm - eUpdatemp)/(4*(epsilon**2))

        return H_OrbOrb, H_CICI, H_OrbCI[idx,:]

    def get_energy(self, h1e, eri, dm1_cas, dm2_cas):
        dm1 = self.CASRDM1_to_RDM1(dm1_cas)
        dm2 = self.CASRDM2_to_RDM2(dm1_cas,dm2_cas)
        E = np.einsum('pq,pq', h1e, dm1) + np.einsum('pqrs,pqrs', eri, dm2)
        return E

    def get_energy_cas(self, h1eff, h2eff, dm1_cas, dm2_cas):
        E = np.einsum('pq,pq', h1eff, dm1_cas) + np.einsum('pqrs,pqrs', h2eff, dm2_cas)
        return E

    def spin_square(self,fcivec):
        return self.fcisolver.spin_square(fcivec,self.ncas,self.nelecas)

    def get_index(self):
        hess = self.get_hessian()

        eigenvalue = scipy.linalg.eigvals(hess)

        eigenvalue = np.around(eigenvalue.real,7)

        eigenvalue_neg = [ev for ev in eigenvalue if ev < 0]
        eigenvalue_pos = [ev for ev in eigenvalue if ev > 0]

        index_neg = len(eigenvalue_neg)
        index_pos = len(eigenvalue_pos)
        nb_zero = len(eigenvalue) - index_neg - index_pos

        return index_neg, index_pos, nb_zero

    def kernel(self):
        ''' This method runs the iterative Newton-Raphson loop '''
        return kernel(self)

    def grid_search(self):
        return grid_search(self)


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

    def read_config(file):
        f = open(file,"r")
        lines = f.read().splitlines()
        basis, charge, spin, cas = 'sto-3g', 0, 0, (0,0)
        for line in lines:
            if re.match('basis', line) is not None:
                basis = re.split(r'\s', line)[-1]
            elif re.match('charge', line) is not None:
                charge = int(re.split(r'\s', line)[-1])
            elif re.match('spin', line) is not None:
                spin = int(re.split(r'\s', line)[-1])
            elif re.match('cas', line) is not None:
                tmp = list(re.split(r'\s', line)[-1])
                cas = (int(tmp[1]), int(tmp[3]))
        return basis, charge, spin, cas

    mol = gto.M(atom=sys.argv[1])
    basis, charge, spin, cas = read_config(sys.argv[2])
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    myhf = mol.RHF().run()
    mycas = NR_CASSCF(myhf,cas[0],cas[1])
    grid_search(mycas)

