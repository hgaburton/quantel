#!/usr/bin/env python

import sys, re
# HGAB TODO: We need to add the mcscf library for now... would be nice to set things up properly
sys.path.append('../../')
import numpy as np
import scipy
from pyscf import gto, ao2mo
from pygnme import wick, utils, owndata
from csf import csf 
from opt.mode_controlling import ModeControl

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

    # Get an initial HF solution
    mf = mol.RHF().run()
    nmo, nocc = mf.mo_occ.size, np.sum(mf.mo_occ > 0)

    # Get overlap matrix
    ovlp = owndata(mf.get_ovlp())
    # Core Hamiltonian
    h1e  = owndata(mf.get_hcore())
    # ERIs in AO basis
    h2e  = owndata(ao2mo.restore(1, mf._eri, mol.nao).reshape(mol.nao**2, mol.nao**2))

    # Load the CI vec data
    ci0  = np.reshape(np.fromfile('bo0_civec.npy')[-400:],(20,20))
    ci1  = np.reshape(np.fromfile('bo1_civec.npy')[-36:],(6,6))
    ci2x = np.reshape(np.fromfile('bo2x_civec.npy')[-4:],(2,2))
    ci2y = np.reshape(np.fromfile('bo2y_civec.npy')[-4:],(2,2))
    ci3  = np.array([[1]])

    # Load the orbital coefficient data
    mo0  = np.genfromtxt('bo0.mo_coeff')
    mo1  = np.genfromtxt('bo1.mo_coeff')
    mo2x = np.genfromtxt('bo2x.mo_coeff')
    mo2y = np.genfromtxt('bo2y.mo_coeff')
    mo3  = np.genfromtxt('bo3.mo_coeff')

    # Reoptimise BO-0
    mc   = csf(mol,0,6,6,0,[0,1,2,3],[4,5,6,7,8,9],"+++---",None,"site")
    mc.initialise(mo0)
    opt  = ModeControl(minstep=0.0, rtrust=0.01)
    opt.run(mc, thresh=1e-10, maxit=500, index=None)
    mo0  = mc.mo_coeff
    del mc

    # Reoptimise BO-1
    mc   = csf(mol,0,4,4,0,[0,1,2,3,4],[5,6,7,8],"++--",None,"site")
    mc.initialise(mo1)
    opt  = ModeControl(minstep=0.0, rtrust=0.01)
    opt.run(mc, thresh=1e-10, maxit=500, index=None)
    mo1  = mc.mo_coeff
    del mc

    # Reoptimise BO-2x
    mc   = csf(mol,0,2,2,0,[0,1,2,3,4,5],[6,7],"+-",None,"site")
    mc.initialise(mo2x)
    opt  = ModeControl(minstep=0.0, rtrust=0.01)
    opt.run(mc, thresh=1e-10, maxit=500, index=None)
    mo2x  = mc.mo_coeff
    del mc

    # Reoptimise BO-2y
    mc   = csf(mol,0,2,2,0,[0,1,2,3,4,5],[6,7],"+-",None,"site")
    mc.initialise(mo2y)
    opt  = ModeControl(minstep=0.0, rtrust=0.01)
    opt.run(mc, thresh=1e-10, maxit=500, index=None)
    mo2y  = mc.mo_coeff
    del mc

    # Save the data in a format that makes CARMA happy... (See Oliver's note) 
    ci   = (owndata(ci3), owndata(ci2x), owndata(ci2y), owndata(ci1),owndata(ci0))
    mo   = (owndata(mo3), owndata(mo2x), owndata(mo2y), owndata(mo1), owndata(mo0))

    # Save information about active orbital spaces for each state
    nact  = (0,2,2,4,6)
    ncore = (7,6,6,5,4)

    # Intialise memory
    nstate = len(ci)
    h = np.zeros((nstate, nstate))
    s = np.zeros((nstate, nstate))

    # Threshold for cutting off CI contributions
    thresh = 1e-10

    # Compute coupling terms
    for x in range(nstate):
        for w in range(x, nstate):
            # Setup biorthogonalised orbital pair
            refx = wick.reference_state[float](nmo, nmo, nocc, nact[x], ncore[x], mo[x])
            refw = wick.reference_state[float](nmo, nmo, nocc, nact[w], ncore[w], mo[w])

            # Setup paired orbitals
            orbs = wick.wick_orbitals[float, float](refx, refw, ovlp)

            # Setup matrix builder object
            mb = wick.wick_rscf[float, float, float](orbs, mol.energy_nuc())
            # Add one- and two-body contributions
            mb.add_one_body(h1e)
            mb.add_two_body(h2e)

            # Generate lists of FCI bitsets
            vx = utils.fci_bitset_list(nocc-ncore[x], nact[x])
            vw = utils.fci_bitset_list(nocc-ncore[w], nact[w])

            # Loop over FCI occupation strings
            for iwa in range(len(vw)):
                for iwb in range(len(vw)):
                    if(abs(ci[w][iwa,iwb]) < thresh):
                        # Skip if coefficient is below threshold
                        continue
                    for ixa in range(len(vx)):
                        for ixb in range(len(vx)):
                            if(abs(ci[x][ixa,ixb]) < thresh):
                                # Skip if coefficient is below threshold
                                continue
                            # Compute S and H contribution for this pair of determinants
                            stmp, htmp = mb.evaluate(vx[ixa], vx[ixb], vw[iwa], vw[iwb])
                            # Accumulate the Hamiltonian and overlap matrix elements
                            h[x,w] += htmp * ci[w][iwa,iwb] * ci[x][ixa,ixb]
                            s[x,w] += stmp * ci[w][iwa,iwb] * ci[x][ixa,ixb]

            h[w,x] = h[x,w]
            s[w,x] = s[x,w]
 
    # Report our result
    print("\n Hamiltonian")
    print(h)
    print("\n Overlap")
    print(s)

    # Solve the generalised eigenvalue problem
    w, v = scipy.linalg.eigh(h, b=s)
    print("\n Recoupled NO-CSF-CI eigenvalues")
    print(w)
    print("\n Recoupled NO-CSF-CI eigenvectors")
    print(v)
