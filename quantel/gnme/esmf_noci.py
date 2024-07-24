#!/usr/bin/python

import numpy as np
from pygnme import wick, utils, owndata


def esmf_coupling(esmf1, esmf2, metric, hcore=None, eri=None, enuc=0.0, thresh=1e-10, with_ref=True):
    # Convert integral matrices to pygnme-friendly format
    ovlp = owndata(metric)

    # Number of orbitals
    assert(esmf1.nmo == esmf1.nmo) 
    nmo = esmf1.nmo

    # Number of electrons 
    assert(esmf1.nalfa == esmf2.nalfa)
    assert(esmf1.nbeta == esmf2.nbeta)
    # TODO: For now, we can only do ms = 0
    assert(esmf1.nalfa == esmf1.nbeta)

    nocc = esmf1.nalfa
    nvir = nmo - nocc

    # Get access to CI coefficients
    if(with_ref):
        cref1 = esmf1.mat_ci[0,0]
        cref2 = esmf2.mat_ci[0,0]
        t1   = 1/np.sqrt(2) * np.reshape(esmf1.mat_ci[1:,0], (esmf1.nalfa, esmf1.nmo - esmf1.nalfa))
        t2   = 1/np.sqrt(2) * np.reshape(esmf2.mat_ci[1:,0], (esmf2.nalfa, esmf2.nmo - esmf2.nalfa))
    else:
        cref1 = 0.0
        cref2 = 0.0
        t1   = 1/np.sqrt(2) * np.reshape(esmf1.mat_ci[:,0], (esmf1.nalfa, esmf1.nmo - esmf1.nalfa))
        t2   = 1/np.sqrt(2) * np.reshape(esmf2.mat_ci[:,0], (esmf2.nalfa, esmf2.nmo - esmf2.nalfa))

    # Intialise output
    Hxw, Sxw = 0, 0

    # Setup biorthogonalised orbital pair
    refx = wick.reference_state[float](nmo, nmo, nocc, owndata(esmf1.mo_coeff))
    refw = wick.reference_state[float](nmo, nmo, nocc, owndata(esmf2.mo_coeff))

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
    Hxw += htmp * cref1 * cref2
    Sxw += stmp * cref1 * cref2

    # Ref-single contribution
    for i in range(nocc):
        for a in range(nvir):
            bx = utils.bitset(refx.m_bs)
            bx.flip(i) 
            bx.flip(a+nocc)
            # Parity
            px = refx.m_bs.parity(bx) 

            # <ia|...|0>
            stmp, htmp = mb.evaluate(bx, refx.m_bs, refw.m_bs, refw.m_bs)
            Hxw += 2.0 * htmp * t1[i,a] * cref2 * px 
            Sxw += 2.0 * stmp * t1[i,a] * cref2 * px

            # <0|...|ia>
            stmp, htmp = mb.evaluate(refx.m_bs, refx.m_bs, bx, refw.m_bs)
            Hxw += 2.0 * htmp * cref1 * t2[i,a] * px
            Sxw += 2.0 * stmp * cref1 * t2[i,a] * px

    # single-single contribution
    for i in range(nocc):
        for a in range(nvir):
            bx = utils.bitset(refx.m_bs)
            bx.flip(i) 
            bx.flip(a+nocc)
            # Parity
            px = refx.m_bs.parity(bx) 

            for j in range(nocc):
                for b in range(nvir):
                    bw = utils.bitset(refw.m_bs)
                    bw.flip(j) 
                    bw.flip(b+nocc)
                    # Parity
                    pw = refw.m_bs.parity(bw) 

                    # same spin
                    stmp, htmp = mb.evaluate(bx, refx.m_bs, bw, refw.m_bs)
                    Hxw += 2.0 * htmp * t1[i,a] * t2[j,b] * px * pw
                    Sxw += 2.0 * stmp * t1[i,a] * t2[j,b] * px * pw

                    # <0|...|ia>
                    stmp, htmp = mb.evaluate(bx, refx.m_bs, refw.m_bs, bw)
                    Hxw += 2.0 * htmp * t1[i,a] * t2[j,b] * px * pw
                    Sxw += 2.0 * stmp * t1[i,a] * t2[j,b] * px * pw

    return Sxw, Hxw
