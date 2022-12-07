r"""
This implements the `search` function for CSFs
"""
#!/usr/bin/env python

import sys, re, os
import numpy as np
from scipy.linalg import expm as scipy_expm
from scipy.linalg import eigvalsh as scipy_eigvalsh
from pyscf import gto
#from ss_casscf import ss_casscf
from mcscf.src.csfs.CSFConstructor import CSFConstructor
from opt.eigenvector_following import EigenFollow


def random_rot(n, lmin, lmax):
    X = lmin + np.random.rand(n, n) * (lmax - lmin)
    X = np.tril(X) - np.tril(X).T
    return scipy_expm(X)


##### Main #####
if __name__ == '__main__':

    np.set_printoptions(linewidth=10000)


    def read_config(file):
        f = open(file, "r")
        lines = f.read().splitlines()
        basis, charge, spin, frozen, cas, grid_option, Hind, maxit, thresh = 'sto-3g', 0, 0, 0, (
        0, 0), 1000, None, 1000, 1e-8
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
                tmp = re.split(r'\s', line)[-1]
                tmp2 = tmp[1:-1].split(',')
                cas = (int(tmp2[0]), int(tmp2[1]))
            elif re.match('nsample', line) is not None:
                nsample = int(re.split(r'\s', line)[-1])
            elif re.match('units', line) is not None:
                unit_str = str(re.split(r'\s', line)[-1])
            elif re.match('thresh', line) is not None:
                thresh = np.power(0.1, int(re.split(r'\s', line)[-1]))
        return basis, charge, spin, frozen, cas, nsample, Hind, maxit, unit_str, thresh


    # Initialise the molecular structure
    basis, charge, spin, frozen, cas, nsample, Hind, maxit, unit_str, thresh = read_config(sys.argv[2])
    mol = gto.Mole(symmetry=False, unit=unit_str)
    mol.atom = sys.argv[1]
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    # Get overlap matrix
    g = mol.intor('int1e_ovlp')

    # Get an initial HF solution
    myhf = mol.RHF().run()

    # Initialise CSF object
    #mycas = ss_casscf(mol, cas[0], cas[1])
    mycsf = CSFConstructor(mol, s, permutation, mo_basis)

    nmo = myhf.mo_coeff.shape[1]
    ndet = mycas.nDet

    # Get inital coefficients and CI vectors
    ref_mo = myhf.mo_coeff.copy()
    ref_ci = np.identity(ndet)

    cas_list = []

    count = 0
    for itest in range(nsample):
        # Randomly perturb CI and MO coefficients
        mo_guess = ref_mo.dot(random_rot(nmo, -np.pi, np.pi))
        ci_guess = ref_ci.dot(random_rot(ndet, -np.pi, np.pi))

        # Set orbital coefficients
        mycas.initialise(mo_guess, ci_guess)
        mycas.canonicalize_()

        opt = EigenFollow(minstep=0.0, rtrust=0.15)
        if not opt.run(mycas, thresh=thresh, maxit=500, index=Hind):
            continue
        hindices = mycas.get_hessian_index()
        pushoff = 0.01
        pushit = 0
        while hindices[0] != Hind and pushit < 5:
            # Try to perturb along relevant number of downhill directions
            mycas.pushoff(1, pushoff)
            opt.run(mycas, thresh=thresh, maxit=500, index=Hind)
            hindices = mycas.get_hessian_index()
            pushoff *= 2
            pushit += 1

        if hindices[0] != Hind: continue

        mycas.canonicalize_()

        # Get the distances
        new = True
        for othercas in cas_list:
            if 1.0 - abs(mycas.overlap(othercas)) < 1e-8:
                new = False
                break
        if new:
            count += 1
            tag = "{:04d}".format(count)
            np.savetxt(tag + '.mo_coeff', mycas.mo_coeff, fmt="% 20.16f")
            np.savetxt(tag + '.mat_ci', mycas.mat_ci, fmt="% 20.16f")
            np.savetxt(tag + '.energy', np.array([
                [mycas.energy, hindices[0], hindices[1], mycas.s2]]), fmt="% 18.12f % 5d % 5d % 12.6f")

            # Deallocate integrals to reduce memory footprint
            #    mycas.deallocate()
            cas_list.append(mycas.copy())
