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
from qland.wfn.csf import csf # This is the csf object we interface with
from qland.opt.eigenvector_following import EigenFollow


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
        #mo_guess = ref_mo.copy()
        #mo_guess[:,2:6] = ref_mo[:,2:6].dot(random_rot(nmo-6, -np.pi, np.pi))
        #print(mo_guess)
        #mo_guess = ref_mo.dot(random_rot(nmo, -np.pi, np.pi))
        mo_guess = ref_mo
        # Set orbital coefficients
        del mycsf
        mo_guess = np.zeros((nmo,nmo))
        mo_guess[0,0] = 1
        mo_guess[5,0] = 1
        mo_guess[0,1] = 1
        mo_guess[5,1] = -1
        mo_guess[1,2] = 1
        mo_guess[6,2] = 1
        mo_guess[1,3] = 1
        mo_guess[6,3] = -1
        mo_guess[4,4] = 1
        mo_guess[9,4] = -1
        mo_guess[3,5] = 1
        mo_guess[8,5] = 1

        mo_guess[2,6] = 1
        mo_guess[7,6] = 0
        mo_guess[2,7] = 0
        mo_guess[7,7] = 1

        mo_guess[3,8] = 1
        mo_guess[8,8] = -1
        mo_guess[4,9] = 1
        mo_guess[9,9] = 1
        mo_guess *= 1./np.sqrt(2)
        mo_guess = ref_mo
        mycsf = csf(mol, spin, cas[0], cas[1], frozen, core, active, g_coupling, permutation, mo_basis)
        mycsf.initialise(mo_guess)

        # Test
        #print("F_core: \n", mycsf.F_core)
        #print("F cas: \n", mycsf.F_cas)
        #num_grad = mycsf.get_numerical_gradient(eps=1e-6)
        #grad = mycsf.gradient
        #grads = np.zeros((grad.shape[0], 2))
        #grads[:, 0] = num_grad
        #grads[:, 1] = grad
        #print(np.round(grads, 5))
        #print("Dimensions: ", grads.shape[0])
        #print(mycsf.rot_idx)
        #num_hess = mycsf.get_numerical_hessian(eps=1e-5)
        #hess = mycsf.hessian
        #print("Numerical Hessian")
        #print(num_hess)
        #np.save("num_hess", num_hess)
        #print("Hessian")
        #print(hess)
        #np.save("hess", hess)
        #num_hess = np.load("num_hess.npy")
        #print("Hessian")
        #print(np.linalg.eigvalsh(num_hess))
        #print(np.linalg.eigvalsh(hess))
        #print(np.allclose(num_hess, hess, rtol=0, atol=1e-5))
        #a = num_hess - hess
        #for i in range(hess.shape[1]):
        #    print(f"{i} {np.allclose(a[:, i], np.zeros(a[:, i].shape), rtol=0, atol=1e-5)}")
        #print(mycsf.ncore)
        #print(mycsf.ncas)
        #print(mycsf.rot_idx)
        #print(np.linalg.eigvalsh(np.load("hess.npy")))
        #quit()
        #print(mycsf.energy)
        #print(np.linalg.eigvalsh(mycsf.hessian))
        #mycsf.canonicalise()
        #print(mycsf.mo_coeff)
        #print(mycsf.energy)
        #print(np.linalg.eigvalsh(mycsf.hessian))

        opt = EigenFollow(minstep=0.0, rtrust=0.15)
        if not opt.run(mycsf, thresh=thresh, maxit=500, index=Hind):
            continue
        hindices = mycsf.get_hessian_index()
 
        #hess = mycsf.hessian
        #print(hess.shape)
        #e,v = np.linalg.eigh(hess)
        #print(mycsf.mo_coeff)
        #X = np.zeros((mycsf.ncas,mycsf.ncas))
        #X[mycsf.rot_idx] = v[:,0]
        #print(X) 
        #
        #print("Eigvalsh")
        #print(np.linalg.eigvalsh(mycsf.hessian))


        #pushoff = 0.01
        #pushit = 0
        #while hindices[0] != Hind and pushit < 0:
        #    break
        #    # Try to perturb along relevant number of downhill directions
        #    mycsf.pushoff(1, pushoff)
        #    opt.run(mycsf, thresh=thresh, maxit=500, index=Hind)
        #    hindices = mycsf.get_hessian_index()
        #    pushoff *= 2
        #    pushit += 1


        #if hindices[0] != Hind: continue
        #continue
        #mycsf.canonicalise()
        #print(mycsf.mo_coeff)
        #print(np.linalg.eigvalsh(mycsf.hessian))
        #tmp = np.zeros((nmo,nmo))
        #tmp[mycsf.rot_idx] = np.linalg.eigh(mycsf.hessian)[1][:,1]
        #print(tmp)
        # Check if energy is still correct
        #mycsf.initialise(mycsf.mo_coeff)
        #if hindices[0] != Hind:
        #    continue
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
        #hess = mycsf.hessian
        #evals, evecs = np.linalg.eigh(hess)
        #colvecs = np.zeros((38, 10))
        #count=0
        #print(mycsf.mo_coeff)
        #for i, ev in enumerate(evals):
        #    if np.isclose(ev, 0, rtol=0, atol=1e-6):
        #        X = np.zeros(mycsf.rot_idx.shape)
        #        X[mycsf.rot_idx] = evecs[:,i]
        #        print()
        #        print(X)
        #        colvecs[:, count] = evecs[:, i]
        #        count += 1
        #print(np.round(colvecs, decimals=3))
        #print(mycsf.rot_idx)
        #print(mycsf.dm1_cas)
        #quit()
    #num_hess = mycsf.get_numerical_hessian(eps=1e-5)
    #hess = mycsf.hessian
    #print("Numerical Hessian")
    #print(num_hess)
    #print("Hessian")
    #print(hess)
    #print("Hessian")
    #print(np.linalg.eigvalsh(num_hess))
    #print(np.linalg.eigvalsh(hess))
    #print(np.allclose(num_hess, hess, rtol=0, atol=1e-5))
    #quit()
    #for i, csf in enumerate(cas_list):
    #    print(csf.csf_info.get_csf_one_rdm_aobas())
        
