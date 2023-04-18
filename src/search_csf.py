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
from csf import csf # This is the csf object we interface with
from opt.eigenvector_following import EigenFollow


def random_rot(n, lmin, lmax):
    X = lmin + np.random.rand(n, n) * (lmax - lmin)
    X = np.tril(X) - np.tril(X).T
    return scipy_expm(X)


##### Main #####
if __name__ == '__main__':

    np.set_printoptions(linewidth=10000, precision=10, suppress=True)


    def read_config(file):
        f = open(file, "r")
        lines = f.read().splitlines()
        basis, charge, spin, frozen, cas, grid_option, Hind, maxit, thresh, core, active, g_coupling, permutation, mo_basis = 'sto-3g', 0, 0, 0, (
        0, 0), 1000, None, 1000, 1e-8, [], [], None, None, 'site'
        nsample = 1
        unit_str = 'A'
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
            elif re.match('core', line) is not None:
                core = [int(x) for x in re.split(r'\s', line)[1:]]
            elif re.match('active', line) is not None:
                active = [int(x) for x in re.split(r'\s', line)[1:]]
            elif re.match('g_coupling', line) is not None:
                g_coupling = line.split()[-1]
            elif re.match('permutation', line) is not None:
                permutation = [int(x) for x in re.split(r'\s', line)[1:]]
            elif re.match('mo_basis', line) is not None:
                mo_basis = re.split(r'\s', line)[-1]
        return basis, charge, spin, frozen, cas, nsample, Hind, maxit, unit_str, thresh, core, active, g_coupling, permutation, mo_basis


    # Initialise the molecular structure
    basis, charge, spin, frozen, cas, nsample, Hind, maxit, unit_str, thresh, core, active, g_coupling, permutation, mo_basis = read_config(sys.argv[2])
    mol = gto.Mole(symmetry=False, unit=unit_str)
    mol.atom = sys.argv[1]
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    # Get overlap matrix
    g = mol.intor('int1e_ovlp')

    # Get an initial HF solution
    # myhf = mol.RHF().run()

    # Initialise CSF object

    mycsf = csf(mol, spin, cas[0], cas[1], frozen, core, active, g_coupling, permutation, mo_basis)
    mycsf.initialise()

    nmo = mycsf.mo_coeff.shape[1]
    ndet = mycsf.nDet

    # Get inital coefficients and CI vectors
    ref_mo = mycsf.csf_info.coeffs.copy()
    ref_ci = np.identity(ndet)

    cas_list = []

    count = 0
    for itest in range(nsample):
        # Randomly perturb CI and MO coefficients
        mo_guess = ref_mo.dot(random_rot(nmo, -np.pi, np.pi))
        # Set orbital coefficients
        del mycsf
        mycsf = csf(mol, spin, cas[0], cas[1], frozen, core, active, g_coupling, permutation, mo_basis)
        mycsf.initialise(mo_guess)

        # Test
        #num_hess = mycsf.get_numerical_hessian(eps=1e-4)
        #hess = mycsf.hessian
        #print("Numerical Hessian")
        #print(num_hess)
        #print("Hessian")
        #print(hess)
        #print("Hessian")
        #print(np.linalg.eigvalsh(num_hess))
        #print(np.linalg.eigvalsh(hess))
        #quit()
        #mycsf.canonicalize_()

        opt = EigenFollow(minstep=0.0, rtrust=0.15)
        if not opt.run(mycsf, thresh=thresh, maxit=500, index=Hind):
            continue
        hindices = mycsf.get_hessian_index()
 
        #hess = mycsf.hessian
        #e,v = np.linalg.eigh(hess)
        #print(mycsf.mo_coeff)
        #print(e)
        #print(v)
        #X = np.zeros((mycsf.ncas,mycsf.ncas))
        #X[mycsf.rot_idx] = v[:,0]
        #print(X) 
        # 
        #print(np.linalg.eigvalsh(mycsf.hessian))


        pushoff = 0.01
        pushit = 0
        while hindices[0] != Hind and pushit < 0:
            # Try to perturb along relevant number of downhill directions
            mycsf.pushoff(1, pushoff)
            opt.run(mycsf, thresh=thresh, maxit=500, index=Hind)
            hindices = mycsf.get_hessian_index()
            pushoff *= 2
            pushit += 1
        #continue
        if hindices[0] != Hind:
            continue
        # Get the distances
        new = True
        for othercas in cas_list:
            o = abs(mycsf.overlap(othercas))
            print("Overlap: ", o)
            if abs(1.0 - o) < 1e-8:
                new = False
                break
        if new:
            print("New CSF found")
            count += 1
            tag = "{:04d}".format(count)
            np.savetxt(tag + '.mo_coeff', mycsf.mo_coeff, fmt="% 20.16f")
            np.savetxt(tag + '.energy', np.array([
                [mycsf.energy, hindices[0], hindices[1], 0.0]]), fmt="% 18.12f % 5d % 5d % 12.6f")
            # Deallocate integrals to reduce memory footprint
            #    mycsf.deallocate()
            cas_list.append(mycsf.copy())
    for i, csf in enumerate(cas_list):
        print(csf.csf_info.get_csf_one_rdm_aobas())
        
