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
from pyscf import fci
from pyscf.mcscf import casci


class NR_CASSCF(casci.CASCI):
    __doc__ = casci.CASCI.__doc__ + '''NR_CASSCF

    '''


##### Main #####
if __name__ == '__main__':

    mol = gto.M(
    atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],],
    basis = '6-31g')
    mol.verbose = 3

    #A simple HF calculation
    myhf = scf.RHF(mol)
    myhf.kernel()

    norb=len(myhf.mo_coeff)
    print('This is the number of orbitals',norb)
    nelec=mol.nelec
    print('This is the number of orbitals',nelec)
    mo = myhf.mo_coeff

    # 4 orbitals, 4 electrons CASCI calculation
    mycas = myhf.CASCI(4, 4)
    mycas.fcisolver.nroots = 3
    mycas.kernel()

    print('This is the number of cas orbitals', mycas.ncas)
    print('This is the number of cas electrons', mycas.nelecas)

    print('This is the solver that we use',mycas.fcisolver)

    #These are the three ci vectors
    #TODO how are they representing CI vectors: here 6x6 matrix
    civec=mycas.ci
    print('These are the CAS ci vectors',civec)

    # 1pdm in AO representation
    dm1 = mycas.make_rdm1(None,mycas.ci[0],4,4)
    #print('This is the 1-RDM matrix', dm1)
    casdm1 = mycas.fcisolver.make_rdm1(mycas.ci[0],4,4)
    print('This is the CAS 1-RDM matrix', casdm1)
    # alpha and beta 1-pdm in AO representation
    dm1_alpha, dm1_beta = mycas.make_rdm1s(None,mycas.ci[0],4,4)

    #transition density matrices in the CAS space
    cast_dm1, cast_dm2 =mycas.fcisolver.trans_rdm12(mycas.ci[0],mycas.ci[1],4,4)
    print('This is the first transition density matrix IN THE CAS SPACE', cast_dm1)

    #we transform the transition density matrices to full MO space (because pySCF gives them in CAS space)
    ncore = mycas.ncore
    ncas = mycas.ncas
    mocore = mo[:,:ncore]
    mocas = mo[:,ncore:ncore+ncas]
    #the occocc part od dm1 is a diagonal of 2
    ao_cast_dm1 = np.einsum('pi,ij,qj->pq', mocas, cast_dm1, mocas)
    t_dm1 = np.dot(mocore, mocore.conj().T) * 2
    t_dm1 = t_dm1 + ao_cast_dm1

    print(cast_dm1.shape)

    #the t_dm
    def dm2_mo_occ(part,p,q,r,s,dm1):
        if part=='ijkl':
            return(2*np.kron(p,q)*np.kron(r,s) - np.kron(p,r)*np.kron(q,s))
        elif part=='ijpq':
            return(np.kron(p,q)*dm1[r,s] - np.kron(p,s)*np.kron(q,r))
        elif part=='pijq':
            return(2*np.kron(q,p)*np.kron(r,s) - 0.5*np.kron(q,r)*dm1[p,q])
        else:
            return('Wrong specification of part')

    dm2 = np.zeros((norb,norb,norb,norb))

    # we use the formula (62)-(67) of the notes to compute the rest of the t_dm2
    # really not elegant ... find an alternative way to do this
    for i in range(ncore):
        for j in range(ncore):
            for k in range(ncore):
                for l in range(ncore):
                    dm2[i,j,k,l]=dm2_mo_occ('ijkl',i,j,k,l,cast_dm1)
    for i in range(ncore):
        for j in range(ncore):
            for p in range(ncas):
                for q in range(ncas):
                    dm2[i,j,p+ncore,q+ncore]=dm2_mo_occ('ijpq',i,j,p,q,cast_dm1)
    for i in range(ncore):
        for j in range(ncore):
            for p in range(ncas):
                for q in range(ncas):
                    dm2[p+ncore,i,j,q+ncore]=dm2_mo_occ('pijq',p,i,j,q,cast_dm1)

    dm2[ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas]=cast_dm2

    #We want to try to contract the above transition density matrix with the eri
    eri = mol.intor('int2e')
    print(eri.shape)

    erimo=np.asarray(mol.ao2mo(mo)) #ao2mo.kernel(mol, mo) gives the same result
    print(np.asarray(mol.ao2mo(mo)).shape)

    eriunfold = ao2mo.restore(1, erimo, norb)
    print(eriunfold.shape)

    #we use the einsum function to perform the actual contraction
    elem2e=np.einsum('ijkl,ijkl',eriunfold,dm2)
    print(elem2e)

    #we also need to contract the one electron integral with
    h1e=mycas.get_hcore()
    elem1e=np.einsum('pq,qp',h1e,t_dm1)
    print(elem1e)