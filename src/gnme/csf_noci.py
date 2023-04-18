r"""
Runs NOCI with CSFs here
"""
import sys, re
sys.path.append('../')
import numpy as np
import scipy
from pyscf import gto, ao2mo
from pygnme import wick, utils, owndata
from csf import csf
from opt.mode_controlling import ModeControl

def csf_proj(mol, nmo, nocc, ovlp, h1e, h2e, ci, mo, ncore, nact, thresh=1e-10):
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

    # Solve the generalised eigenvalue problem
    w, v = scipy.linalg.eigh(h, b=s)
    return h, s, w, v
