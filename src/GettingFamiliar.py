#!/usr/bin/env python
# Author: Antoine Marie

import sys

from functools import reduce
import numpy as np
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



##### Main #####
if __name__ == '__main__':

#Definition of the molecule
    mol = gto.M(
    atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],],
    basis = '6-31g')
    mol.verbose = 3
    #We can also fix the spin and the symmetry in this object

    nelec=mol.nelec #Access the number of electrons
    print('This is the number of orbitals',nelec) # (5,5)

    v1e = mol.intor('int1e_nuc') #Nuclear repulsion matrix elements
    t1e = mol.intor('int1e_kin') #Kinetic energy matrix elements
    h1e_ao = v1e + t1e

    eri_ao = mol.intor('int2e') #eri in the ao basis in the chemist notation
    print(eri_ao.shape) # (13,13,13,13)

    norb = eri_ao.shape[0] #We can access the number of orbital like this
    print('This is the number of orbitals',norb) # 13

# We start by running a simple HF calculation
    myhf = scf.RHF(mol)
    myhf.kernel() #Run the calculation

    mo = myhf.mo_coeff #Obtain the coefficient of the MO

    #myhf.analyze() we can use this command to obtain the Mulliken charge analysis, ...

    # Now we transform the 2-electrons integral to the MO basis
    eri=np.asarray(mol.ao2mo(mo)) #ao2mo.kernel(mol, mo) gives the same result
    print(np.asarray(mol.ao2mo(mo)).shape)

    eri = ao2mo.restore(1, eri, norb)
    print(eri.shape)

    h1e = np.einsum('ip,ij,jq->pq', mo, h1e_ao, mo) # We also transform the 1-electron integrals

# Now we have a look at the CASCI object on which are built the CASSCF object
    mycas = myhf.CASCI(4, 4) # 4 orbitals, 4 electrons
    mycas.fcisolver.nroots = 3 # Choose the number of roots for the Davidson procedure
    mycas.kernel()

    ncore = mycas.ncore # Number of core orbitals
    ncas = mycas.ncas # Number of cas orbitals
    nvir = norb - ncore - ncas # Number of virtual orbitals
    print('This is the number of core ',ncore,', active ',ncas,', and virtual orbitals', nvir)

    nelecas = mycas.nelecas # Number of active electrons
    print('This is the number of cas electrons', mycas.nelecas)

    print('This is the solver that we use',mycas.fcisolver) # We can choose to change this solver

    civec = mycas.ci # List of the three ci vectors
    print('These are the CAS ci vectors',civec)
    print('Shape of a ci vector', civec[0].shape) # (6,6) Row (resp. column) correspond to a given alpha (resp. beta) config, 6=binom(4 2)

    dm1 = mycas.make_rdm1(None,mycas.ci[0],4,4) # 1pdm in AO representation, shape (13,13)
    #print('This is the 1-RDM matrix', dm1)
    dm1_alpha, dm1_beta = mycas.make_rdm1s(None,mycas.ci[0],4,4) # alpha and beta 1-pdm in AO representation
    dm1_cas = mycas.fcisolver.make_rdm1(mycas.ci[0],4,4) # 1 pdm of the active space in MO representation, shape (4,4)
    #print('This is the CAS 1-RDM matrix', casdm1)

    t_dm1_cas, t_dm2_cas =mycas.fcisolver.trans_rdm12(mycas.ci[0],mycas.ci[1],4,4) # transition density matrices in the CAS space, shape (4,4) and (4,4,4,4)
    print('This is the first transition density matrix in the CAS space', t_dm1_cas)

#Now we want to compute matrix elements of the CI gradient

    # We need to transform the transition density matrices to full MO space
    mo_core = mo[:,:ncore]
    mo_cas = mo[:,ncore:ncore+ncas]

    t_dm1 = np.zeros((norb,norb))
    t_dm1_core = 2*np.diag(np.ones(ncore)) #the occocc part of dm1 is a diagonal of 2
    t_dm1[:ncore,:ncore] = t_dm1_core
    t_dm1[ncore:ncore+ncas,ncore:ncore+ncas] = t_dm1_cas

    #We can obtain the AO representation as
    t_dm1_cas_ao = np.einsum('pi,ij,qj->pq', mo_cas, t_dm1_cas, mo_cas)
    t_dm1_ao = np.einsum('pi,ij,qj->pq', mo, t_dm1, mo)

    # The ijkl, ijpq and pijq can be simplified according to the following function (Eq(62)-(67)).
    # The piqj is zero as all the elements with a virtual index or an odd number of occupied indices.
    def dm2_mo_occ(part,p,q,r,s,dm1):
        if part=='ijkl':
            return(2*np.kron(p,q)*np.kron(r,s) - np.kron(p,r)*np.kron(q,s))
        elif part=='ijpq':
            return(np.kron(p,q)*dm1[r,s] - np.kron(p,s)*np.kron(q,r))
        elif part=='pijq':
            return(2*np.kron(q,p)*np.kron(r,s) - 0.5*np.kron(q,r)*dm1[p,q])
        else:
            return('Wrong specification of part')

    t_dm2 = np.zeros((norb,norb,norb,norb))

    for i in range(ncore):      # really not elegant ... find an alternative way to do this
        for j in range(ncore):
            for k in range(ncore):
                for l in range(ncore):
                    t_dm2[i,j,k,l]=dm2_mo_occ('ijkl',i,j,k,l,t_dm1_cas)
    for i in range(ncore):
        for j in range(ncore):
            for p in range(ncas):
                for q in range(ncas):
                    t_dm2[i,j,p+ncore,q+ncore]=dm2_mo_occ('ijpq',i,j,p,q,t_dm1_cas)
    for i in range(ncore):
        for j in range(ncore):
            for p in range(ncas):
                for q in range(ncas):
                    t_dm2[p+ncore,i,j,q+ncore]=dm2_mo_occ('pijq',p,i,j,q,t_dm1_cas)

    t_dm2[ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas] = t_dm2_cas # Finally we add the uvxy sector

    # To compute the CI gradient we need to contract h1e
    CIgrad_1e = np.einsum('pq,qp', h1e, t_dm1) #TODO verify definition of h1e and t_dm1, (pq or qp)

    CIgrad_2e = np.einsum('ijkl,ijkl', eri, t_dm2) #TODO same as above

#Now we want to try to compute elements of the orbital gradient
    GenF = mycas.get_fock(mo,mycas.ci[0],eri,dm1_cas)
    print(GenF)
    print(GenF.shape)
    # This function gives the generalized Fock matrix, however to compute the orbital gradient
    # we would like to separate the Fock matrix into a core and an active part

    def get_F_core(norb,ncore,h1e,eri): #TODO we could probably optimize this...
        F_core = np.zeros((norb,norb))
        F_core += h1e
        for p in range(norb):
            for q in range(norb):
                for i in range(ncore):
                    F_core[p,q] += 2*eri[p,q,i,i] - eri[p,i,i,q]
        return(F_core)

    def get_F_cas(norb,ncas,dm1_cas,eri): #TODO same here
        F_cas = np.zeros((norb,norb))
        for p in range(norb):
            for q in range(norb):
                for t in range(ncas):
                    for u in range(ncas):
                        F_cas[p,q] += dm1_cas[t,u]*(eri[p,q,t,u]-0.5*eri[p,u,t,q])
        return(F_cas)

# We want to know how to compute the S^2 value of a given wave function
    S = spin_op.spin_square0(mycas.ci[0],ncas,nelecas) #This function should be used only for RHF CI wave function
    print(S) # Singlet multiplicity one
    S = spin_op.spin_square0(mycas.ci[1],ncas,nelecas)
    print(S) # Triplet multiplicity three

