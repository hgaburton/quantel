#!/usr/bin/python

import numpy as np
from pygnme import wick, utils, owndata


def pcid_coupling(wfn1, wfn2, metric, hcore=None, eri=None, enuc=0.0, thresh=1e-10):
    # Convert integral matrices to pygnme-friendly format
    ovlp = owndata(metric)

    # Number of orbitals
    assert(wfn1.nmo == wfn1.nmo) 
    nmo = wfn1.nmo

    # Number of electrons 
    assert(wfn1.nelec[0] == wfn2.nelec[0])
    assert(wfn1.nelec[1] == wfn2.nelec[1])
    # TODO: For now, we can only do ms = 0
    assert(wfn1.nelec[0] == wfn1.nelec[1])

    nocc = wfn1.nelec[0]
    nvir = nmo - nocc

    # Get access to CI coefficients
    cref1 = wfn1.mat_ci[0,0]
    cref2 = wfn2.mat_ci[0,0]
    t1   = np.reshape(wfn1.mat_ci[1:,0], (wfn1.na, wfn1.nmo - wfn1.na))
    t2   = np.reshape(wfn2.mat_ci[1:,0], (wfn2.na, wfn2.nmo - wfn2.na))

    # Intialise output
    Hwx, Swx = 0, 0

    # Setup biorthogonalised orbital pair
    refx = wick.reference_state[float](nmo, nmo, nocc, owndata(wfn1.mo_coeff))
    refw = wick.reference_state[float](nmo, nmo, nocc, owndata(wfn2.mo_coeff))

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

    # Ref-Ref contribution
    stmp, htmp = mb.evaluate(refx.m_bs, refx.m_bs, refw.m_bs, refw.m_bs)
    Hwx += htmp * cref1 * cref2
    Swx += stmp * cref1 * cref2

    # Ref-single contribution
    for i in range(nocc):
        for a in range(nvir):
            bx = utils.bitset(refx.m_bs)
            bx.flip(i) 
            bx.flip(a+nocc)

            # <ia|...|0>
            stmp, htmp = mb.evaluate(bx, bx, refw.m_bs, refw.m_bs)
            Hwx += htmp * t1[i,a] * cref2 
            Swx += stmp * t1[i,a] * cref2

            # <0|...|ia>
            stmp, htmp = mb.evaluate(refx.m_bs, refx.m_bs, bx, bx)
            Hwx += htmp * cref1 * t2[i,a]
            Swx += stmp * cref1 * t2[i,a]

    # single-single contribution
    for i in range(nocc):
        for a in range(nvir):
            bx = utils.bitset(refx.m_bs)
            bx.flip(i) 
            bx.flip(a+nocc)

            for j in range(nocc):
                for b in range(nvir):
                    bw = utils.bitset(refw.m_bs)
                    bw.flip(j) 
                    bw.flip(b+nocc)

                    # same spin
                    stmp, htmp = mb.evaluate(bx, bx, bw, bw)
                    Hwx += htmp * t1[i,a] * t2[j,b]
                    Swx += stmp * t1[i,a] * t2[j,b]

    return Swx, Hwx
