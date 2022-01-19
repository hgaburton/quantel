#!/usr/bin/env python
# Author: Antoine Marie

import sys
import re
import numpy as np
import datetime

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

    tmp_mol = re.split(r'\s', lines[4])
    tmp_cas = re.split(r'\s', lines[5])
    tmp_det = re.split(r'\s', lines[6])

    nelec = int(tmp_mol[4])
    norb = int(tmp_mol[-4])
    ncas = int(tmp_cas[6])
    nelecas = int(tmp_cas[-5])
    nDet = int(tmp_det[-3])

    return norb, nelec, ncas, nelecas, nDet


def get_coeff(file): #TODO get coeff -> MO coeff, CAS DM, CI coeff
    f = open(file,"r")
    lines = f.read().splitlines()

    norb, nelec, ncas, nelecas, nDet = get_size(file)

    NO, CI, DM1 = [], [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match('This is the CI', line) is not None:
            tmp = []
            for j in range(nDet):
                list_str = np.array(re.split(r'\s+', lines[i+2+j]))
                empty_str = np.array([''])
                list_str = np.setdiff1d(list_str,empty_str,True)
                tmp.append([float(k) for k in list_str])
            CI.append(tmp)
            i += nDet + 1
        if re.match('This is the diagonal', line) is not None:
            tmp = []
            for j in range(ncas):
                list_str = np.array(re.split(r'\s+', lines[i+2+j]))
                empty_str = np.array([''])
                list_str = np.setdiff1d(list_str,empty_str,True)
                tmp.append([float(k) for k in list_str])
            DM1.append(tmp)
            i += ncas + 1
        if re.match('This is the natural', line) is not None:
            tmp = []
            for j in range(norb):
                list_str = np.array(re.split(r'\s+', lines[i+2+j]))
                empty_str = np.array([''])
                list_str = np.setdiff1d(list_str,empty_str,True)
                tmp.append([float(k) for k in list_str])
            NO.append(tmp)
            i += norb + 1
        i+=1
    return NO, CI, DM1

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


##### Main #####
if __name__ == '__main__':
    file = sys.argv[1]

    nrj, index, spin, nb_it, time_tot = get_results(file)

    unique_sols = select_unique(concatenate([nrj,index,spin]))
    print("There are ", len(unique_sols), " unique solutions.")
    print(['NRJ','Nb neg','Nb pos','Nb zero','Spin','Mul'])
    matprint(unique_sols)

    print('The total calculation time is ', datetime.timedelta(seconds=time_tot))

    print(get_size(file))

    get_coeff(file)