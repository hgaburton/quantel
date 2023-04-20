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
    if(type(nocc) is int):
        na = nocc
        nb = nocc
    else:
        na, nb = nocc[0], nocc[1]

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
            refxa = wick.reference_state[float](nmo, nmo, na, nact[x], ncore[x], mo[x])
            refxb = wick.reference_state[float](nmo, nmo, nb, nact[x], ncore[x], mo[x])
            refwa = wick.reference_state[float](nmo, nmo, na, nact[w], ncore[w], mo[w])
            refwb = wick.reference_state[float](nmo, nmo, nb, nact[w], ncore[w], mo[w])
            
            # Setup paired orbitals
            orba = wick.wick_orbitals[float, float](refxa, refwa, ovlp)
            orbb = wick.wick_orbitals[float, float](refxb, refwb, ovlp)

            # Setup matrix builder object
            mb = wick.wick_uscf[float, float, float](orba, orbb, mol.energy_nuc())
            # Add one- and two-body contributions
            mb.add_one_body(h1e)
            mb.add_two_body(h2e)

            # Generate lists of FCI bitsets
            vxa = utils.fci_bitset_list(na-ncore[x], nact[x])
            vxb = utils.fci_bitset_list(nb-ncore[x], nact[x])
            vwa = utils.fci_bitset_list(na-ncore[w], nact[w])
            vwb = utils.fci_bitset_list(nb-ncore[w], nact[w])

            # Loop over FCI occupation strings
            for iwa in range(len(vwa)):
                for iwb in range(len(vwb)):
                    if(abs(ci[w][iwa,iwb]) < thresh):
                        # Skip if coefficient is below threshold
                        continue
                    for ixa in range(len(vxa)):
                        for ixb in range(len(vxb)):
                            if(abs(ci[x][ixa,ixb]) < thresh):
                                # Skip if coefficient is below threshold
                                continue
                            # Compute S and H contribution for this pair of determinants
                            stmp, htmp = mb.evaluate(vxa[ixa], vxb[ixb], vwa[iwa], vwb[iwb])
                            # Accumulate the Hamiltonian and overlap matrix elements
                            h[x,w] += htmp * ci[w][iwa,iwb] * ci[x][ixa,ixb]
                            s[x,w] += stmp * ci[w][iwa,iwb] * ci[x][ixa,ixb]

            h[w,x] = h[x,w]
            s[w,x] = s[x,w]

    # HGAB: You can use the LibGNME eigensolver to avoid issues with singular overlap matrix
    #       Just be aware that it returns eigval as a matrix not an array 
    #       Here, the threshold corresponds to the null space of the overlap matrix
    eigval, v = utils.gen_eig_sym(nstate, h, s,thresh=1e-8)
    w = eigval[0,:]
    # Solve the generalised eigenvalue problem
    #w, v = scipy.linalg.eigh(h, b=s)
    return h, s, w, v
