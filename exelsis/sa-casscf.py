#!/usr/bin/env python
# Author: Antoine Marie

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1


import sys
from functools import reduce
import numpy as np
import scipy.linalg
import re
import datetime
import random
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
from pyscf.lo import orth
from pyscf.fci.cistring import make_strings, parity
from gnme import utils

if __name__ == '__main__':


    def read_config(file):
        f = open(file,"r")
        lines = f.read().splitlines()
        basis, charge, spin, frozen, cas, grid_option, Hind, maxit = 'sto-3g', 0, 0, 0, (0,0), 1000, None, 1000
        for line in lines:
            if re.match('basis', line) is not None:
                basis = str(re.split(r'\s', line)[-1])
            elif re.match('charge', line) is not None:
                charge = int(re.split(r'\s', line)[-1])
            elif re.match('spin', line) is not None:
                spin = int(re.split(r'\s', line)[-1])
            elif re.match('frozen', line) is not None:
                frozen = int(re.split(r'\s', line)[-1])
            elif re.match('seed', line) is not None:
                random.seed(int(line.split()[-1]))
            elif re.match('index', line) is not None:
                Hind = int(line.split()[-1])
            elif re.match('maxit', line) is not None:
                maxit = int(line.split()[-1])
            elif re.match('active_space', line) is not None:
                tmp = re.split(r'\s+', line)
                cas = (int(tmp[-2]), int(tmp[-1]))
            elif re.match('grid', line) is not None:
                if re.split(r'\s', line)[-1] == 'full':
                    grid_option = re.split(r'\s', line)[-1]
                else:
                    grid_option = int(re.split(r'\s', line)[-1])
        return basis, charge, spin, frozen, cas, grid_option, Hind, maxit

    # initialise stuff
    mol = gto.Mole(symmetry=False,unit='B')
    mol.atom = sys.argv[1]
    basis, charge, spin, frozen, cas, grid_option, Hind, maxit = read_config(sys.argv[2])
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()
    metric = mol.intor("int1e_ovlp")

    mf = scf.RHF(mol)
    mf.kernel()
    mf.mo_coeff[:,[8,12]] = mf.mo_coeff[:,[12,8]]
    
    w=np.ones((4,))
    mc = mcscf.state_average_(mcscf.CASSCF(mf, cas[0], cas[1],), (w/np.sum(w)).tolist())
    mc.verbose = 4 
    mc.kernel()
    mo = mc.mo_coeff

    ci = np.array([x.ravel() for x in mc.ci]).T


    emc = mc.casci(mo)[0]

    print('E(CAS) = %.12f, ref = -75.982521066893' % emc)

    for i in range(4):
        tag="{:04d}".format(i+1)
        cisave = ci.copy()
        cisave[:,[0,i]] = ci[:,[i,0]]

        np.savetxt(tag+'.mo_coeff', mo, fmt="% 20.16f")
        np.savetxt(tag+'.mat_ci',   cisave, fmt="% 20.16f")
        np.savetxt(tag+'.energy',   
                   np.array([[0, 0, 0, 0]]),
                   fmt="% 18.12f % 5d % 5d % 12.6f")
