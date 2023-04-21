#!/usr/bin/env python

import sys, re, os
import numpy as np
from scipy.linalg import expm as scipy_expm
from scipy.linalg import eigvalsh as scipy_eigvalsh
from pyscf import gto
from xesla.wfn.esmf import ESMF
from xesla.opt.eigenvector_following import EigenFollow

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
                tmp = re.split(r'\s', line)[-1]
                tmp2 = tmp[1:-1].split(',')
                cas = (int(tmp2[0]), int(tmp2[1]))
            elif re.match('nsample', line) is not None:
                nsample = int(re.split(r'\s', line)[-1])
            elif re.match('units', line) is not None:
                unit_str = str(re.split(r'\s', line)[-1])
            elif re.match('thresh', line) is not None:
                thresh = np.power(0.1,int(re.split(r'\s', line)[-1]))
        return basis, charge, spin, frozen, cas, nsample, Hind, maxit, unit_str, thresh

    # Initialise the molecular structure
    basis, charge, spin, frozen, cas, nsample, Hind, maxit, unit_str, thresh = read_config(sys.argv[2])
    mol = gto.Mole(symmetry=False,unit=unit_str)
    mol.atom = sys.argv[1]
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    # Get an initial HF solution
    myhf = mol.RHF().run()

    # Initialise wfn object
    myfun = ESMF(mol)
    nmo = myhf.mo_coeff.shape[1]
    ndet = myfun.nDet

    # Get inital coefficients and CI vectors
    ref_mo = myhf.mo_coeff.copy()
    ref_ci = np.identity(ndet)
    
    half_rot = scipy_expm(0.25*np.pi*np.matrix([[0,-1],[1,0]]))

    sol_list = []

    count = 0
    for itest in range(nsample):
        # Randomly perturb CI and MO coefficients
        mo_guess = ref_mo.copy()
#        mo_guess[:,[2,3]] = mo_guess[:,[2,3]].dot(half_rot)
        mo_guess = ref_mo.dot(random_rot(nmo,  -np.pi, np.pi))
        ci_guess = ref_ci.dot(random_rot(ndet, -np.pi, np.pi))

        # Set orbital coefficients
        del myfun
        myfun = ESMF(mol)
        myfun.initialise(mo_guess, ci_guess)
        #num_hess = myfun.get_numerical_hessian(eps=1e-4)
        #hess = myfun.hessian
        #print("Numerical Hessian")
        #print(num_hess)
        #print("Hessian")
        #print(hess)
        #print("Hessian")
        #print(np.linalg.eigvalsh(num_hess))
        #print(np.linalg.eigvalsh(hess))
        #quit()

        #mycas.canonicalize_()

        opt = EigenFollow(minstep=0.0,rtrust=0.15)
        if not opt.run(myfun, thresh=thresh, maxit=500, index=Hind):
            continue
        hindices = myfun.get_hessian_index()
        myfun.update_integrals()
        #pushoff = 0.01
        #pushit  = 0
        #while hindices[0] != Hind and pushit < 5: 
        #    # Try to perturb along relevant number of downhill directions
        #    mycas.pushoff(1,pushoff)
        #    opt.run(mycas, thresh=thresh, maxit=500, index=Hind)
        #    hindices = mycas.get_hessian_index()
        #    pushoff *= 2
        #    pushit  += 1

        if hindices[0] != Hind: continue
        
        # Get the distances
        new = True
        for othercas in sol_list:
            if 1.0 - abs(myfun.overlap(othercas)) < 1e-8:
                new = False
                break
        print(new)
        if new: 
            count += 1
            tag = "{:04d}".format(count)
            np.savetxt(tag+'.mo_coeff', myfun.mo_coeff, fmt="% 20.16f")
            np.savetxt(tag+'.mat_ci', myfun.mat_ci, fmt="% 20.16f")
            np.savetxt(tag+'.energy', np.array([
                  [myfun.energy, hindices[0], hindices[1], 0.0]]), fmt="% 18.12f % 5d % 5d % 12.6f")
            
            # Deallocate integrals to reduce memory footprint
            myfun.deallocate()
            sol_list.append(myfun.copy())
