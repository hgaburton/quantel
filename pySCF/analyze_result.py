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

def get_NR_info(file,nb):
    """ Read result.txt file and return the information of the calculation nb """
    f = open(file,"r")
    lines = f.read().splitlines()

    norb, nelec, ncas, nelecas, nDet = get_size(file)

    nrj, index, spin, nb_it, time_tot = None, None, None, None, None

    MO, CI, NO, DM1, nrjMO, CIno = None, None, None, None, None, None

    i = 0
    while i < len(lines):
        line = lines[i]

        if re.match('Start the Newton-Raphson calculation number '+str(int(nb))+' ', line) is not None:

            k = 1
            tmp_line = lines[i+k]

            nrj, index, spin, nb_it, time_tot = None, None, None, None, None
            warn = False

            while re.match('Start the Newton-Raphson calculation number' , tmp_line) is None:
                if re.match('The energy', tmp_line) is not None:
                    nrj = np.around(float((re.split(r'\s', tmp_line))[-2]),7)
                if re.match('The hessian of this', tmp_line) is not None:
                    tmp = (re.split(r'\s', tmp_line))
                    index = (int(tmp[7]), int(tmp[12]), int(tmp[18]))
                if re.match('The squared spin', tmp_line) is not None:
                    tmp = (re.split(r'\s', tmp_line))
                    spin = (np.around(float(tmp[10]),4), np.around(float(tmp[18]),4))
                if re.match('The Newton-Raphson has converged', tmp_line) is not None:
                    nb_it = int(re.split(r'\s', tmp_line)[-3])
                if re.match('This grid calculation took', tmp_line) is not None:
                    time_tot = float(re.split(r'\s', tmp_line)[-3])
                if re.match('This is the CI coefficients', tmp_line) is not None:
                    tmp = []
                    for j in range(nDet):
                        list_str = np.array(re.split(r'\s+', lines[i+k+2+j]))
                        empty_str = np.array([''])
                        list_str = np.setdiff1d(list_str,empty_str,True)
                        tmp.append([float(k) for k in list_str])
                    CI = tmp
                    k += nDet + 1
                if re.match('The occupations', tmp_line) is not None:
                    tmp = []
                    for j in range(ncas):
                        list_str = np.array(re.split(r'\s+', lines[i+k+2+j]))
                        empty_str = np.array([''])
                        list_str = np.setdiff1d(list_str,empty_str,True)
                        tmp.append([float(k) for k in list_str])
                    DM1 = tmp
                    k += 3
                if re.match('This is the natural', tmp_line) is not None:
                    tmp = []
                    for j in range(norb):
                        list_str = np.array(re.split(r'\s+', lines[i+k+2+j]))
                        empty_str = np.array([''])
                        list_str = np.setdiff1d(list_str,empty_str,True)
                        tmp.append([float(k) for k in list_str])
                    NO = tmp
                    k += norb + 1
                if re.match('This is the MO', tmp_line) is not None:
                    tmp = []
                    for j in range(norb):
                        list_str = np.array(re.split(r'\s+', lines[i+k+2+j]))
                        empty_str = np.array([''])
                        list_str = np.setdiff1d(list_str,empty_str,True)
                        tmp.append([float(k) for k in list_str])
                    MO = tmp
                    k += norb + 1
                if re.match('This is the CI vector', tmp_line) is not None:
                    tmp = []
                    list_str = np.array(re.split(r'\s+', lines[i+k+2]))
                    empty_str = np.array([''])
                    list_str = np.setdiff1d(list_str,empty_str,True)
                    tmp.append([float(k) for k in list_str])
                    CIno = tmp
                    k += 3
                if re.match('The energies of the ', tmp_line) is not None:
                    tmp = []
                    list_str = np.array(re.split(r'\s+', lines[i+k+2]))
                    empty_str = np.array([''])
                    list_str = np.setdiff1d(list_str,empty_str,True)
                    tmp.append([float(k) for k in list_str])
                    nrjMO = tmp
                    k += 3
                if re.match('WARN', tmp_line) is not None:
                    warn = True

                k+=1
                tmp_line = lines[i+k]

            break # We have found the desired Newton-Raphson calculation
        i+=1

    if i==len(lines):
        print("This calculation id does not exist")

    return nrj, index, spin, nb_it, time_tot, MO, CI, NO, DM1, nrj, CIno, warn

def scalar_twosol(file,myhf,nbsol1,nbsol2):
    """ Compute the scalar product between two solutions given their number in the file """
    norb, nelec, ncas, nelecas, nDet = get_size(file)

    nrj1, index1, spin1, nb_it1, time_tot1, MO1, CI1, NO1, DM11, nrj1, CIno1, warn1 = get_NR_info(file,nbsol1)
    nrj2, index2, spin2, nb_it2, time_tot2, MO2, CI2, NO2, DM12, nrj2, CIno2, warn2 = get_NR_info(file,nbsol2)

    mycas1 = NR_CASSCF(myhf,ncas,nelecas,thresh=1e-7)
    mycas1._initMO = np.array(MO1)
    mycas1._initCI = np.array(CI1)
    mycas1.initializeMO()
    mycas1.initializeCI()

    mycas2 = NR_CASSCF(myhf,ncas,nelecas,thresh=1e-7)
    mycas2._initMO = np.array(MO2)
    mycas2._initCI = np.array(CI2)
    mycas2.initializeMO()
    mycas2.initializeCI()

    metric = mycas1.mol.intor('int1e_ovlp')

    # print("Projecting CAS_2 into active space for CAS_1:")
    projvec = cas_proj(mycas1, mycas2, metric)
    # print(projvec)

    # print("Total overlap = {:20.10f}".format(mycas_new.mat_CI[:,0].dot(projvec)))
    scal = mycas1.mat_CI[:,0].dot(projvec)
    return scal

def get_results(file):
    f = open(file,"r")
    lines = f.read().splitlines()

    id_calc, nrj, index, spin, nb_it = [], [], [], [], []
    time_tot = None

    for line in lines:
        if re.match('Start the Newton-Raphson calculation number', line) is not None:
            id_calc.append(int(re.split(r'\s', line)[-2]))
        if re.match('The Newton-Raphson has not', line) is not None:
            id_calc.pop(-1)
        if re.match('The energy', line) is not None:
            nrj.append(np.around(float((re.split(r'\s', line))[-2]),7))
        if re.match('The hessian of this', line) is not None:
            tmp = (re.split(r'\s', line))
            index.append((int(tmp[7]), int(tmp[12]), int(tmp[18])))
        if re.match('The squared spin', line) is not None:
            tmp = (re.split(r'\s', line))
            spin.append((np.around(float(tmp[10]),4), np.around(float(tmp[18]),4)))
        if re.match('The Newton-Raphson has converged', line) is not None:
            nb_it.append(int(re.split(r'\s', line)[-3]))
        if re.match('This grid calculation took', line) is not None:
            time_tot = float(re.split(r'\s', line)[-3])
    return id_calc, nrj, index, spin, nb_it, time_tot

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
    """ Compute the scalar product between the different solutions to search for the number of different wave functions of a given energy. """
    id_calc, nrj, index, spin, nb_it, time_tot = get_results(file)
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
                vecWF[pos].append((MO[i],CI[i]))
            else:
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

                    if np.around(scal,4) == 1: #and np.around(scal_det,4) == 1:
                        break

                    if j == len(vecWF[pos])-1:
                        vecWF[pos].append((MO[i],CI[i]))

                    j+=1

    return vecWF




##### Main #####
if __name__ == '__main__':
    file = sys.argv[1]

    id_calc, nrj, index, spin, nb_it, time_tot = get_results(file)
    unique_sols = np.unique(concatenate([nrj,index,spin]),axis=0)

    numbersol = []
    for sol in unique_sols:
        numbersol.append(id_calc[concatenate([nrj,index,spin]).tolist().index(sol.tolist())])

    unique_sols = concatenate([numbersol,unique_sols])

    print("There are ", len(unique_sols), " unique solutions.")
    print(["Nb calc",'NRJ','Nb neg','Nb pos','Nb zero','Spin','Mul'])
    matprint(unique_sols)

    print('The total calculation time is ', datetime.timedelta(seconds=time_tot))

    # We can utilize this to plot the index as a function of the nrj
    # print(unique_sols[:,(0,1)])

    # NO, MO, CI, DM1 = get_coeff(file)

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

    scal = scalar_twosol(file,myhf,1,35)
    print(scal)

    dupe = []
    n = len(unique_sols)
    # n = 10
    for i in range(n):
        for j in range(i+1,n):
            if np.around(unique_sols[i,1],4) == np.around(unique_sols[j,1],4):
                scal = scalar_twosol(file,myhf,unique_sols[i,0],unique_sols[j,0])
                if np.around(scal,4) == 1:
                    dupe.append(j)
    print(dupe)
    print(np.unique(dupe))
    if len(np.unique(dupe))>0:
        unique_sols = np.delete(unique_sols,np.unique(dupe),axis=0)
    print("There are ", len(unique_sols), " unique solutions.")
    print(["Nb calc",'NRJ','Nb neg','Nb pos','Nb zero','Spin','Mul'])
    matprint(unique_sols[:10])
