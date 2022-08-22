#!/usr/bin/env python
# Author: Antoine Marie

import os
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
from ss_casscf import ss_casscf
from newton_raphson import NewtonRaphson
from lbfgs import LBFGS
from itertools import cycle

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

    np.set_printoptions(linewidth=10000,precision=6,suppress=True)

    # initialise stuff
    basis, charge, spin, frozen, cas, grid_option, Hind, maxit = read_config(sys.argv[2])
    mol = gto.Mole(symmetry=False,unit='B',charge=charge,spin=spin,basis=basis)
    mol.atom = sys.argv[1]
    mol.build()
    myhf = mol.RHF().run()
    np.savetxt('hf_mo_energy',np.vstack((np.arange(myhf.mo_coeff.shape[1])+1,myhf.mo_energy,myhf.mo_occ)).T,
                           fmt=["%5d","%16.10f","%16.10f"])
    np.savetxt('hf_mo_coeff',myhf.mo_coeff,fmt="%16.10f")

    # Run a reference calculation
    myhf.mo_coeff[:,[6,8,9,12]] = myhf.mo_coeff[:,[8,6,12,9]]
    #mc = mcscf.state_average_(mcscf.CASSCF(myhf, 2, 2,), [1.0,0.0,0.0,0.0])
    #mc.verbose = 4
    #mc.kernel()
    #mo_guess = mc.mo_coeff
    #ci_guess = np.vstack([np.ravel(v) for v in mc.ci]).T
 
    #mo_guess = np.genfromtxt('mo_coeff')
    #ci_guess = np.genfromtxt('mat_ci')

    molist = []
    cilist = []
    mycas = ss_casscf(mol,cas[0],cas[1])

    mo_guess = myhf.mo_coeff.copy()
    mo_guess[:,[6,8,9,12]] = mo_guess[:,[8,6,12,9]]
    ci_guess = np.identity(4)

    print(mo_guess)
    print(ci_guess)
    mycas.initialise(mo_guess, ci_guess[:,:])
    LBFGS(mycas,plev=1)
    molist.append([mycas.mo_coeff.copy()])
    cilist.append([mycas.mat_CI.copy()])

    hess = mycas.get_hessian()
    e, v = np.linalg.eigh(hess)
    print("Hessian index = ", np.sum(e < 0))

    mycas.canonicalize_()
    np.savetxt('mo_energy',np.vstack((np.arange(mycas.nmo)+1,mycas.mo_energy,mycas.mo_occ)).T,
                           fmt=["%5d","%16.10f","%16.10f"])
    np.savetxt('mo_coeff',mycas.mo_coeff,fmt="%16.10f")
    np.savetxt('mat_ci',mycas.mat_CI,fmt="%16.10f")

    mycas.initialise(mo_guess, ci_guess)
    NewtonRaphson(mycas,plev=1)
    hess = mycas.get_hessian()
    e, v = np.linalg.eigh(hess)
    print("Hessian index = ", np.sum(e < 0))

    ## Now work through bond angles
    #ref_geom = np.copy(mol.atom_coords())
    #print()
    #for theta in np.linspace(0,np.pi,361):
    #    coords = ref_geom.copy()
    #    coords[0:2,1] =   np.cos(theta) * ref_geom[0:2,1] + np.sin(theta) * ref_geom[0:2,2]
    #    coords[0:2,2] = - np.sin(theta) * ref_geom[0:2,1] - np.cos(theta) * ref_geom[0:2,2]
    #    mol.set_geom_(coords)
    #    

    #    elist = [theta*180/np.pi]
    #    for i in range(4):
    #        mycas = ss_casscf(mol,cas[0],cas[1])
    #        mycas.initialise(molist[i][-1], cilist[i][-1])
    #        LBFGS(mycas,plev=1)
    #        molist[i].append(mycas.mo_coeff.copy())
    #        cilist[i].append(mycas.mat_CI.copy())

    #        elist.append(mycas.energy)

    #    print(" {: 5.2f}  {: 16.10f}  {: 16.10f}  {: 16.10f}  {: 16.10f}".format(*elist))
    #    sys.stdout.flush()
