#!/usr/bin/env python

import sys, re, os, glob
import numpy as np
from scipy.linalg import expm as scipy_expm
from pyscf import gto
from ss_casscf import ss_casscf
from opt.eigenvector_following import EigenFollow
from opt.mode_controlling import ModeControl
from opt.newton_raphson import NewtonRaphson

def random_rot(n, lmin, lmax):
    X = lmin + np.random.rand(n,n) * (lmax - lmin)
    X = np.tril(X)  - np.tril(X).T
    return scipy_expm(X)
    

##### Main #####
if __name__ == '__main__':

    np.set_printoptions(linewidth=10000)

    def read_config(file):
        f = open(file,"r")
        lines = f.read().splitlines()
        basis, charge, spin, frozen, cas, grid_option, Hind, maxit, thresh = 'sto-3g', 0, 0, 0, (0,0), 1000, None, 1000, 1e-8
        nsample = 1
        unit_str = 'B'
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
                np.random.seed(int(line.split()[-1]))
            elif re.match('index', line) is not None:
                Hind = int(line.split()[-1])
            elif re.match('maxit', line) is not None:
                maxit = int(line.split()[-1])
            elif re.match('cas', line) is not None:
                tmp = list(re.split(r'\s', line)[-1])
                cas = (int(tmp[1]), int(tmp[3]))
            elif re.match('nsample', line) is not None:
                nsample = int(re.split(r'\s', line)[-1])
            elif re.match('units', line) is not None:
                unit_str = str(re.split(r'\s', line)[-1])
            elif re.match('thresh', line) is not None:
                thresh = np.power(0.1,int(re.split(r'\s', line)[-1]))
        return basis, charge, spin, frozen, cas, nsample, Hind, maxit, unit_str, thresh

    # Get string for previous states
    prefix_lst = sys.argv[3:]

    # Initialise the molecular structure
    basis, charge, spin, frozen, cas, nsample, Hind, maxit, unit_str, thresh = read_config(sys.argv[2])
    mol = gto.Mole(symmetry=False,unit=unit_str)
    mol.atom = sys.argv[1]
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    # Get overlap matrix
    g = mol.intor('int1e_ovlp')

    # Get an initial HF solution
    myhf = mol.RHF().run()

    # Initialise CAS object
    mycas = ss_casscf(mol, cas[0], cas[1])

    nmo = myhf.mo_coeff.shape[1]
    ndet = mycas.nDet

    # Get inital coefficients and CI vectors
    ref_mo = myhf.mo_coeff.copy()
    ref_ci = np.identity(ndet)

    cas_list = []
    e_list   = []
    i_list   = []

    count = 0
    # Loop over the prefixes
    for prefix in prefix_lst:
        # Need to count the number of states to converge
        nstates = len(glob.glob(prefix+"*.energy"))
        for i in range(nstates):
            old_tag = "{:s}{:04d}".format(prefix, i+1)
            mo_guess = np.genfromtxt(old_tag+".mo_coeff")
            ci_guess = np.genfromtxt(old_tag+".mat_ci")

            # Set orbital coefficients
            mycas.initialise(mo_guess, ci_guess)
            opt = ModeControl(rtrust=0.001,maxstep=0.1,minstep=1e-8)
            if not opt.run(mycas, thresh=thresh, maxit=10000, index=None):
                continue
            s2 = mycas.s2
            hindices = mycas.get_hessian_index()
            print(hindices)
            #mycas.canonicalize_()

            # Get the distances
            new = True
            for prev, othercas in enumerate(cas_list):
                if 1.0 - abs(mycas.overlap(othercas)) < 1e-8:
                    new = False
                    break
            if new: 
                count += 1
                cas_list.append(mycas.copy())
                tag = "{:04d}".format(count)
                np.savetxt(tag+'.mo_coeff', mycas.mo_coeff, fmt="% 20.16f")
                np.savetxt(tag+'.mat_ci', mycas.mat_ci, fmt="% 20.16f")
                np.savetxt(tag+'.energy', np.array([
                      [mycas.energy, hindices[0], hindices[1], s2]]), fmt="% 18.12f % 5d % 5d % 12.6f")
                e_list.append(mycas.energy)
                i_list.append(hindices[0])
            else: 
                print("Solution matches previous solution...",prev+1)

    np.savetxt('energy_list', np.array([e_list]),fmt="% 16.10f")
    np.savetxt('ind_list', np.array([i_list]),fmt="% 5d")

    dcas = len(cas_list)
    dist_mat = np.zeros((dcas,dcas))
    for i, cas_i in enumerate(cas_list):
        for j, cas_j in enumerate(cas_list):
            if(i<j): continue
            dist_mat[i,j] = cas_i.overlap(cas_j) 
            dist_mat[j,i] = dist_mat[i,j]
    np.savetxt('cas_ov', dist_mat, fmt="% 8.6f")
