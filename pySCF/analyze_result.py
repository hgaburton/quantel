#!/usr/bin/env python
# Author: Antoine Marie

import sys
import re
import numpy as np
import datetime
from functools import reduce
from NR_CASSCF import NR_CASSCF, grid_point
from cas_noci import cas_proj
from pyscf import gto
from pyscf.fci.cistring import make_strings, parity

float_formatter = "{:.6f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def get_results(file):
    f = open(file,"r")
    lines = f.read().splitlines()

    nrj, index, spin, nb_it = [], [], [], []
    time_tot = None

    for line in lines:
        if re.match('The energy', line) is not None:
            nrj.append(float((re.split(r'\s', line))[-2]))
        if re.match('The hessian of this', line) is not None:
            tmp = (re.split(r'\s', line))
            index.append((float(tmp[7]), float(tmp[12]), float(tmp[18])))
        if re.match('The squared spin', line) is not None:
            tmp = (re.split(r'\s', line))
            spin.append((np.around(float(tmp[10]),7), np.around(float(tmp[18]),7)))
        if re.match('The Newton', line) is not None:
            nb_it.append(re.split(r'\s', line)[-3])
        if re.match('This grid calculation took', line) is not None:
            time_tot = float(re.split(r'\s', line)[-3])

    return nrj, index, spin, nb_it, time_tot

def get_size(file):
    f = open(file,"r")
    lines = f.read().splitlines()

    for i in range(10):
        line = lines[i]
        if re.match('The molecule has', line) is not None:
            tmp_mol = re.split(r'\s', line)
        if re.match('The active space', line) is not None:
            tmp_cas = re.split(r'\s', line)
        if re.match('Therefore there i', line) is not None:
            tmp_det = re.split(r'\s', line)

    nelec = int(tmp_mol[4])
    norb = int(tmp_mol[-4])
    ncas = int(tmp_cas[6])
    nelecas = int(tmp_cas[-5])
    nDet = int(tmp_det[-3])

    return norb, nelec, ncas, nelecas, nDet


def get_coeff(file):
    f = open(file,"r")
    lines = f.read().splitlines()

    norb, nelec, ncas, nelecas, nDet = get_size(file)

    MO, CI, NO, DM1, nrj, CIno = [], [], [], [], [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match('This is the CI coefficients', line) is not None:
            tmp = []
            for j in range(nDet):
                list_str = np.array(re.split(r'\s+', lines[i+2+j]))
                empty_str = np.array([''])
                list_str = np.setdiff1d(list_str,empty_str,True)
                tmp.append([float(k) for k in list_str])
            CI.append(tmp)
            i += nDet + 1
        if re.match('The occupations', line) is not None:
            tmp = []
            for j in range(ncas):
                list_str = np.array(re.split(r'\s+', lines[i+2+j]))
                empty_str = np.array([''])
                list_str = np.setdiff1d(list_str,empty_str,True)
                tmp.append([float(k) for k in list_str])
            DM1.append(tmp)
            i += 3
        if re.match('This is the natural', line) is not None:
            tmp = []
            for j in range(norb):
                list_str = np.array(re.split(r'\s+', lines[i+2+j]))
                empty_str = np.array([''])
                list_str = np.setdiff1d(list_str,empty_str,True)
                tmp.append([float(k) for k in list_str])
            NO.append(tmp)
            i += norb + 1
        if re.match('This is the MO', line) is not None:
            tmp = []
            for j in range(norb):
                list_str = np.array(re.split(r'\s+', lines[i+2+j]))
                empty_str = np.array([''])
                list_str = np.setdiff1d(list_str,empty_str,True)
                tmp.append([float(k) for k in list_str])
            MO.append(tmp)
            i += norb + 1
        if re.match('This is the CI vector', line) is not None:
            tmp = []
            list_str = np.array(re.split(r'\s+', lines[i+2]))
            empty_str = np.array([''])
            list_str = np.setdiff1d(list_str,empty_str,True)
            tmp.append([float(k) for k in list_str])
            CIno.append(tmp)
            i += 3
        if re.match('The energies of the ', line) is not None:
            tmp = []
            list_str = np.array(re.split(r'\s+', lines[i+2]))
            empty_str = np.array([''])
            list_str = np.setdiff1d(list_str,empty_str,True)
            tmp.append([float(k) for k in list_str])
            nrj.append(tmp)
            i += 3
        i+=1
    return NO, MO, CI, DM1

def concatenate(list):
    tmp = list
    nb_sol = len(list[0])

    for i in range(len(list)):
        tmp[i] = np.reshape(list[i],(nb_sol,-1))

    return np.concatenate(tmp,axis=1)

def select_unique(list,tol=7):
    tmp = np.around(list,tol)
    unique = np.unique(tmp,axis=0)
    return unique

def matprint(mat, fmt="g"):
    if len(np.shape(np.asarray(mat)))==1:
        mat = mat.reshape(1,len(mat))

    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def read_config(file):
    f = open(file,"r")
    lines = f.read().splitlines()
    basis, charge, spin, cas, grid_option = 'sto-3g', 0, 0, (0,0), 1000
    for line in lines:
        if re.match('basis', line) is not None:
            basis = str(re.split(r'\s', line)[-1])
        elif re.match('charge', line) is not None:
            charge = int(re.split(r'\s', line)[-1])
        # elif re.match('spin', line) is not None:
        #     spin = int(re.split(r'\s', line)[-1])
        elif re.match('cas', line) is not None:
            tmp = list(re.split(r'\s', line)[-1])
            cas = (int(tmp[1]), int(tmp[3]))
        elif re.match('grid', line) is not None:
            if re.split(r'\s', line)[-1] == 'full':
                grid_option = re.split(r'\s', line)[-1]
            else:
                grid_option = int(re.split(r'\s', line)[-1])
    return basis, charge, spin, cas, grid_option

def different_wavefunction(file):
    nrj, index, spin, nb_it, time_tot = get_results(file)
    unique_nrj = select_unique(nrj)
    print(unique_nrj)
    NO, MO, CI, DM1 = get_coeff(file)
    norb, nelec, ncas, nelecas, nDet = get_size(file)

    mol = gto.Mole()
    mol.atom = sys.argv[2]
    mol.unit = 'B'
    basis, charge, spin, cas, grid_option = read_config(sys.argv[3])
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()
    myhf = mol.RHF().run()

    metric = mol.intor('int1e_ovlp')

    vecWF = [[i] for i in unique_nrj] # List of list of degenerate wave functions, each sublist has as first element the energy of the following degenerate wf

    for i in range(len(nrj)):
            # print("Newton-Raphson number", i+1)
            tmp_nrj = np.around(nrj[i],7)
            pos = np.where(unique_nrj == tmp_nrj)[0][0]
            if len(vecWF[pos]) == 1:
                # print("First vector with energy ", nrj[i],"\n")
                vecWF[pos].append((MO[i],CI[i]))
            else:
                # print("Already ", len(vecWF[pos])-1, " vector with energy ", nrj[i], ", starting loop on those vectors")
                mycas_new = NR_CASSCF(myhf,ncas,nelecas,thresh=1e-7)
                mycas_new._initMO = np.array(MO[i])
                mycas_new._initCI = np.array(CI[i])
                mycas_new.initializeMO()
                mycas_new.initializeCI()
                j = 1
                length = len(vecWF[pos])
                while j<length:
                    mycas_tmp = NR_CASSCF(myhf,ncas,nelecas,thresh=1e-7)
                    mycas_tmp._initMO = np.array(vecWF[pos][j][0])
                    mycas_tmp._initCI = np.array(vecWF[pos][j][1])
                    mycas_tmp.initializeMO()
                    mycas_tmp.initializeCI()

                    # print("Projecting CAS_2 into active space for CAS_1:")
                    projvec = cas_proj(mycas_new, mycas_tmp, metric)
                    # print(projvec)

                    # print("Total overlap = {:20.10f}".format(mycas_new.mat_CI[:,0].dot(projvec)))
                    scal = mycas_new.mat_CI[:,0].dot(projvec)
                    # print("")

                    # nelec = mycas_tmp.nelec[0] + mycas_tmp.nelec[1]
                    # S = reduce(np.dot,(mycas_new.mo_coeff.T,metric,mycas_tmp.mo_coeff))
                    # print(nelec)
                    # scal_det = np.linalg.det(S)
                    # print(scal_det)

                    if np.around(scal,4) == 1: #and np.around(scal_det,4) == 1:
                        break

                    if j == len(vecWF[pos])-1:
                        vecWF[pos].append((MO[i],CI[i]))
                        # for k in range(1,len(vecWF[pos])):
                        #     matprint(np.array(vecWF[pos][k][0]))
                        #     matprint(np.array(vecWF[pos][k][1]))

                    j+=1

    return vecWF




##### Main #####
if __name__ == '__main__':
    file = sys.argv[1]

    nrj, index, spin, nb_it, time_tot = get_results(file)
    unique_sols = select_unique(concatenate([nrj,index,spin]))
    print("There are ", len(unique_sols), " unique solutions.")
    print(['NRJ','Nb neg','Nb pos','Nb zero','Spin','Mul'])
    matprint(unique_sols)

    print(unique_sols[:,(0,1)])

    print('The total calculation time is ', datetime.timedelta(seconds=time_tot))

    unique_nrj = select_unique(nrj)
    print(unique_nrj)

    norb, nelec, ncas, nelecas, nDet = get_size(file)
    print(get_size(file))

    NO, MO, CI, DM1 = get_coeff(file)

    # for i in DM1:
    #     print(np.around(np.trace(i),5))

    vec = different_wavefunction(file)
    for i in range(len(vec)):
        print("There are ", len(vec[i])-1, "solutions with energy ", unique_nrj[i] )
        for j in range(1,len(vec[i])):
            print("The ",j,"th solution MO are")
            matprint(np.array(vec[i][j][0]))
            print("and the CI coefficients are")
            matprint(np.array(vec[i][j][1]))

    # mol = gto.Mole()
    # mol.atom = sys.argv[2]
    # mol.unit = 'B'
    # basis, charge, spin, cas, grid_option = read_config(sys.argv[3])
    # mol.basis = basis
    # mol.charge = charge
    # mol.spin = spin
    # mol.build()
    # myhf = mol.RHF().run()
    #
    # metric = mol.intor('int1e_ovlp')
    #
    # print((MO[0],np.array(CI[0])[:,0]),(MO[32],np.array(CI[32])[:,0]))
    # transform_casvec((MO[0],np.array(CI[0])[:,0]),(MO[32],np.array(CI[32])[:,0]),metric)