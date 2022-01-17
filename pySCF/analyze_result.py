#!/usr/bin/env python
# Author: Antoine Marie

import sys
import re
import numpy as np

def get_energy(file):
    f = open(sys.argv[1],"r")
    lines = f.read().splitlines()
    nrj = []
    for line in lines:
        if re.match('The energy', line) is not None:
            nrj.append(float((re.split(r'\s', line))[-2]))
    return nrj

def get_index(file):
    f = open(sys.argv[1],"r")
    lines = f.read().splitlines()
    index = []
    for line in lines:
        if re.match('The hessian of this', line) is not None:
            tmp = (re.split(r'\s', line))
            index.append((float(tmp[7]), float(tmp[12]), float(tmp[18])))
    return index

def get_spin(file):
    f = open(sys.argv[1],"r")
    lines = f.read().splitlines()
    spin = []
    for line in lines:
        if re.match('The squared spin', line) is not None:
            tmp = (re.split(r'\s', line))
            spin.append((np.around(float(tmp[10]),7), np.around(float(tmp[18]),7)))
    return spin

def select_unique(list,tol=7):
    tmp = np.around(list,tol)
    unique = np.unique(tmp,axis=0)
    return unique


##### Main #####
if __name__ == '__main__':
    file = sys.argv[1]

    nrj = get_energy(file)

    print(select_unique(nrj))
    print(len(select_unique(nrj)))

    index = get_index(file)

    float_formatter = "{:.6f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    spin = get_spin(file)

    tmp = np.concatenate((np.reshape(nrj,(len(nrj),1)),np.reshape(index,(len(index),3)),np.reshape(spin,(len(spin),2))),axis=1)
    print(['NRJ','Nb pos','Nb neg','Nb zero','Spin','Mul'])
    print(select_unique(tmp))