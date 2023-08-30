#!/usr/bin/env python

import sys, re
import numpy as np
from scipy.linalg import expm as scipy_expm
from pyscf import gto
from ss_casscf import ss_casscf
from opt.eigenvector_following import EigenFollow
from gnme.cas_noci import cas_proj
from pyscf.tools import cubegen

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
        basis, charge, spin, frozen, cas, grid_option, Hind, maxit = 'sto-3g', 0, 0, 0, (0,0), 1000, None, 1000
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
        return basis, charge, spin, frozen, cas, nsample, Hind, maxit, unit_str

    # Initialise the molecular structure
    basis, charge, spin, frozen, cas, nsample, Hind, maxit, unit_str = read_config(sys.argv[2])
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
    mo_guess = np.genfromtxt(sys.argv[3])
    ci_guess = np.genfromtxt(sys.argv[4])

    # Set orbital coefficients
    mycas.initialise(mo_guess, ci_guess)
    mycas.canonicalize_()

    #for i in range(mycas.ncore,mycas.ncore+mycas.ncas):
    for i in range(mycas.ncore+mycas.ncas):
    #for i in range(mycas.norb):
        cubegen.orbital(mol, 'mo.{:d}.cube'.format(i), mycas.mo_coeff[:,i])

    s2 = mycas.s2
    hindices = mycas.get_hessian_index()

    hess = mycas.hessian
    eigval, eigvec = np.linalg.eigh(hess)
    nrot = mycas.nrot 

    print("\n  Energy = {: 16.10f}".format(mycas.energy))
    print("   <S^2> = {: 16.10f}".format(mycas.s2))
    print("   Index = {: 5d}".format(np.sum(eigval<0)))

    print("\n  ----------------------------------------")
    print("  Natural Orbitals:")
    print("  ----------------------------------------")
    for i in range(mycas.ncore+mycas.ncas):
        print(" {:5d}  {: 10.6f}  {: 10.6f}".format(i+1, mycas.mo_occ[i], mycas.mo_energy[i]))
    print("  ----------------------------------------")

    print("\n  ----------------------------------------")
    print("  Hessian eigenvalue structure:")
    print("     n      Eval       |V_scf|    |V_ci| ") 
    print("  ----------------------------------------")
    for i, eigval in enumerate(eigval):
        vec_scf = eigvec[:nrot,i]
        vec_ci  = eigvec[nrot:,i]
        print(" {:5d}  {: 10.6f}  {: 10.6f} {: 10.6f}".format(
              i, eigval, np.linalg.norm(vec_scf), np.linalg.norm(vec_ci)))
    print("  ----------------------------------------")

     
    ci_eig = 0.5 * np.linalg.eigvalsh(hess[nrot:,nrot:]) + mycas.energy
    print("\n  ----------------------------------------")
    print("  CI eigenvalues:")
    print("     n      Eval")
    print("  ----------------------------------------")
    for i in range(ci_eig.size+1):
        if(i==0): print(" {:5d}  {: 10.6f}".format(i, mycas.energy))
        else: print(" {:5d}  {: 10.6f}".format(i, ci_eig[i-1]))
    print("  ----------------------------------------")

    spin_dens = mycas.get_spin_dm1()
    cubegen.density(mol, 'spin.dens.cube', spin_dens)
