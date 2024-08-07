#!/usr/bin/python

import numpy as np
from pygnme import wick, utils, owndata

def cas_coupling(cas1, cas2, metric, hcore=None, eri=None, enuc=0.0, thresh=1e-10):
    # Convert integral matrices to pygnme-friendly format
    ovlp = owndata(metric)

    # Number of orbitals
    assert(cas1.nmo == cas2.nmo) 
    nmo = cas1.nmo

    # Number of electrons 
    assert(cas1.nalfa == cas2.nalfa)
    assert(cas1.nbeta == cas2.nbeta)
    # TODO: For now, we can only do ms = 0
    assert(cas1.nalfa == cas1.nbeta)
    nocc = cas1.nalfa

    # Reshape CI matrices
    ci1 = np.reshape(cas1.mat_ci[:,0],(cas1.ndeta, cas1.ndetb))
    ci2 = np.reshape(cas2.mat_ci[:,0],(cas2.ndeta, cas2.ndetb))

    # Intialise output
    Hxw, Sxw = 0, 0

    # Setup biorthogonalised orbital pair
    refx = wick.reference_state[float](nmo, nmo, nocc, cas1.cas_nmo, cas1.ncore, owndata(cas1.mo_coeff))
    refw = wick.reference_state[float](nmo, nmo, nocc, cas2.cas_nmo, cas2.ncore, owndata(cas2.mo_coeff))

    # Setup paired orbitals
    orbs = wick.wick_orbitals[float, float](refx, refw, ovlp)

    # Setup matrix builder object
    mb = wick.wick_rscf[float, float, float](orbs, enuc)

    # Add one- and two-body contributions
    if(hcore is not None):
        h1e  = owndata(hcore)
        mb.add_one_body(h1e)
    if(eri is not None):
        h2e  = owndata(eri)
        mb.add_two_body(h2e)

    # Generate lists of FCI bitsets
    vx = utils.fci_bitset_list(nocc-cas1.ncore, cas1.cas_nmo)
    vw = utils.fci_bitset_list(nocc-cas2.ncore, cas2.cas_nmo)
    test = np.zeros((4,4))

    # Loop over FCI occupation strings
    for ixa in range(len(vx)):
        for ixb in range(len(vx)):
            if(abs(ci1[ixa,ixb]) < thresh):
                # Skip if coefficient is below threshold
                continue
            for iwa in range(len(vw)):
                for iwb in range(len(vw)):
                    if(abs(ci2[iwa,iwb]) < thresh):
                        # Skip if coefficient is below threshold
                        continue
                    # Compute S and H contribution for this pair of determinants
                    stmp, htmp = mb.evaluate(vx[ixa], vx[ixb], vw[iwa], vw[iwb])
                    # Accumulate the Hamiltonian and overlap matrix elements
                    Hxw += htmp * ci1[ixa,ixb] * ci2[iwa,iwb]
                    Sxw += stmp * ci1[ixa,ixb] * ci2[iwa,iwb]

    return Sxw, Hxw
