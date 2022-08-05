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
from cas_noci import *
from pyscf.fci.cistring import make_strings, parity
from gnme import utils
from NR_CASSCF import *

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
            elif re.match('cas', line) is not None:
                tmp = list(re.split(r'\s', line)[-1])
                cas = (int(tmp[1]), int(tmp[3]))
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
    myhf = mol.RHF().run()

    np.set_printoptions(linewidth=10000)

    mc = mcscf.state_average_(mcscf.CASSCF(myhf, 2, 2,), [0.25,0.25,0.25,0.25])
    mc.verbose = 4
    mc.kernel()
    MOguess = mc.mo_coeff
    CIguess = np.vstack([np.ravel(v) for v in mc.ci]).T

    print(CIguess)


    mycas = NR_CASSCF(myhf,cas[0],cas[1],Hind=Hind,maxit=maxit)
    mycas.initMO
    mycas.initCI
    mycas.initializeMO()
    mycas.initializeCI()
    mycas.initHeff


    newcas = NR_CASSCF(myhf,cas[0],cas[1],initMO=MOguess,initCI=CIguess[:,[0,1,2,3]],Hind=Hind,maxit=maxit)
    newcas.kernel()

    newcas = NR_CASSCF(myhf,cas[0],cas[1],initMO=MOguess,initCI=CIguess[:,[1,2,3,0]],Hind=Hind,maxit=maxit)
    newcas.kernel()

    newcas = NR_CASSCF(myhf,cas[0],cas[1],initMO=MOguess,initCI=CIguess[:,[2,3,0,1]],Hind=Hind,maxit=maxit)
    newcas.kernel()

    newcas = NR_CASSCF(myhf,cas[0],cas[1],initMO=MOguess,initCI=CIguess[:,[3,0,1,2]],Hind=Hind,maxit=maxit)
    newcas.kernel()
