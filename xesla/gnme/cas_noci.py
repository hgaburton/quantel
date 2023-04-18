#!/usr/bin/python

import numpy as np
from pyscf.fci.cistring import make_strings, parity
from pygnme import wick, utils, owndata
from gnme.wick import gnme_pair


def cas_overlap(cas1, cas2, metric, thresh=1e-10):
    # Convert integral matrices to pygnme-friendly format
    ovlp = owndata(metric)

    # Number of orbitals
    assert(cas1.nmo == cas2.nmo) 
    nmo = cas1.nmo

    # Number of electrons 
    assert(cas1.nelec[0] == cas2.nelec[0])
    assert(cas1.nelec[1] == cas2.nelec[1])
    # TODO: For now, we can only do ms = 0
    assert(cas1.nelec[0] == cas1.nelec[1])
    nocc = cas1.nelec[0]

    # Reshape CI matrices
    ci1 = np.reshape(cas1.mat_ci[:,0],(cas1.nDeta, cas1.nDetb))
    ci2 = np.reshape(cas2.mat_ci[:,0],(cas2.nDeta, cas2.nDetb))

    # Intialise output
    Hwx, Swx = 0, 0

    # Setup biorthogonalised orbital pair
    refx = wick.reference_state[float](nmo, nmo, nocc, cas1.ncas, cas1.ncore, owndata(cas1.mo_coeff))
    refw = wick.reference_state[float](nmo, nmo, nocc, cas2.ncas, cas2.ncore, owndata(cas2.mo_coeff))

    # Setup paired orbitals
    orbs = wick.wick_orbitals[float, float](refx, refw, ovlp)

    # Setup matrix builder object
    mb = wick.wick_rscf[float, float, float](orbs)

    # Generate lists of FCI bitsets
    vx = utils.fci_bitset_list(nocc-cas1.ncore, cas1.ncas)
    vw = utils.fci_bitset_list(nocc-cas2.ncore, cas2.ncas)

    # Loop over FCI occupation strings
    for iwa in range(len(vw)):
        for iwb in range(len(vw)):
            if(abs(ci1[iwa,iwb]) < thresh):
                # Skip if coefficient is below threshold
                continue
            for ixa in range(len(vx)):
                for ixb in range(len(vx)):
                    if(abs(ci2[ixa,ixb]) < thresh):
                        # Skip if coefficient is below threshold
                        continue
                    # Compute S and H contribution for this pair of determinants
                    stmp, htmp = mb.evaluate(vx[ixa], vx[ixb], vw[iwa], vw[iwb])
                    # Accumulate the Hamiltonian and overlap matrix elements
                    Hwx += htmp * ci1[iwa,iwb] * ci2[ixa,ixb]
                    Swx += stmp * ci1[iwa,iwb] * ci2[ixa,ixb]

    return Swx


def cas_hamiltonian(cas1, cas2, metric, hcore, eri, enuc, thresh=1e-10):
    # Convert integral matrices to pygnme-friendly format
    ovlp = owndata(metric)
    h1e  = owndata(hcore)
    h2e  = owndata(eri)

    # Number of orbitals
    assert(cas1.nmo == cas2.nmo) 
    nmo = cas1.nmo

    # Number of electrons 
    assert(cas1.nelec[0] == cas2.nelec[0])
    assert(cas1.nelec[1] == cas2.nelec[1])
    # TODO: For now, we can only do ms = 0
    assert(cas1.nelec[0] == cas1.nelec[1])
    nocc = cas1.nelec[0]

    # Reshape CI matrices
    ci1 = np.reshape(cas1.mat_ci[:,0],(cas1.nDeta, cas1.nDetb))
    ci2 = np.reshape(cas2.mat_ci[:,0],(cas2.nDeta, cas2.nDetb))

    # Intialise output
    Hwx, Swx = 0, 0

    # Setup biorthogonalised orbital pair
    refx = wick.reference_state[float](nmo, nmo, nocc, cas1.ncas, cas1.ncore, owndata(cas1.mo_coeff))
    refw = wick.reference_state[float](nmo, nmo, nocc, cas2.ncas, cas2.ncore, owndata(cas2.mo_coeff))

    # Setup paired orbitals
    orbs = wick.wick_orbitals[float, float](refx, refw, ovlp)

    # Setup matrix builder object
    mb = wick.wick_rscf[float, float, float](orbs, enuc)
    # Add one- and two-body contributions
    mb.add_one_body(h1e)
    mb.add_two_body(h2e)

    # Generate lists of FCI bitsets
    vx = utils.fci_bitset_list(nocc-cas1.ncore, cas1.ncas)
    vw = utils.fci_bitset_list(nocc-cas2.ncore, cas2.ncas)

    # Loop over FCI occupation strings
    for iwa in range(len(vw)):
        for iwb in range(len(vw)):
            if(abs(ci1[iwa,iwb]) < thresh):
                # Skip if coefficient is below threshold
                continue
            for ixa in range(len(vx)):
                for ixb in range(len(vx)):
                    if(abs(ci2[ixa,ixb]) < thresh):
                        # Skip if coefficient is below threshold
                        continue
                    # Compute S and H contribution for this pair of determinants
                    stmp, htmp = mb.evaluate(vx[ixa], vx[ixb], vw[iwa], vw[iwb])
                    # Accumulate the Hamiltonian and overlap matrix elements
                    Hwx += htmp * ci1[iwa,iwb] * ci2[ixa,ixb]
                    Swx += stmp * ci1[iwa,iwb] * ci2[ixa,ixb]

    return Swx, Hwx


def cas_proj(cas_x, cas_w, metric):
    '''Project the first CI vector from cas_w into the active space of cas_x'''
    print("cas_proj")

    # Get number of basis funcs, cas orbitals, electrons, and cas electrons
    nbsf   = cas_x.mo_coeff.shape[0]
    na, nb = cas_x.nelec
    ncas   = cas_x.ncas
    na_cas, nb_cas = cas_x.nelecas
    ncore_a  = na - na_cas
    ncore_b  = nb - nb_cas

    # Check the CAS objects are consistent
    assert(nbsf == cas_w.mo_coeff.shape[0])
    assert(na == cas_w.nelec[0])
    assert(nb == cas_w.nelec[1])
    assert(ncas == cas_w.ncas)
    assert(na_cas == cas_w.nelecas[0])
    assert(nb_cas == cas_w.nelecas[1])

    # Setup gnme_pair for alpha and beta
    apair = gnme_pair(cas_x.mo_coeff, cas_w.mo_coeff, ncore_a, na, ncas, metric) 
    bpair = gnme_pair(cas_x.mo_coeff, cas_w.mo_coeff, ncore_b, nb, ncas, metric) 

    # Get reference bit strings
    ba_ref = make_strings(range(ncas), na_cas)[0]
    bb_ref = make_strings(range(ncas), nb_cas)[0]

    # Initialise projected vector
    proj_vec = np.zeros(cas_x.nDet) 
    xind = 0
    for bx_str_a in make_strings(range(ncas),na_cas):
        # Parity of first string
        pxa = parity(ba_ref, bx_str_a)

        for bx_str_b in make_strings(range(ncas),nb_cas):
            # Parity of second string
            pxb = parity(bb_ref, bx_str_b)

            # Get vector of alpha overlaps
            Sa_tmp = np.zeros(cas_x.nDeta)
            for ia, bw_str_a in enumerate(make_strings(range(ncas),na_cas)):
                # Parity of w alpha string
                pwa = parity(ba_ref, bw_str_a)
                # Get alpha overlap
                Sa_tmp[ia] = pwa * pxa * np.prod(apair.get_overlap(bx_str_a,bw_str_a))

            # Get vector of beta overlaps
            Sb_tmp = np.zeros(cas_x.nDetb)
            for ib, bw_str_b in enumerate(make_strings(range(ncas),nb_cas)):
                # Parity of w beta string
                pwb = parity(bb_ref, bw_str_b)
                # Get beta overlap
                Sb_tmp[ib] = pwb * pxb * np.prod(apair.get_overlap(bx_str_b,bw_str_b))

            # Evaluate this component
            proj_vec[xind] = np.dot(np.outer(Sa_tmp,Sb_tmp).ravel(), cas_w.mat_ci[:,0])
            
            # Increment counter
            xind += 1

    return proj_vec

if __name__=='__main__':
    from pyscf import gto
    from NR_CASSCF import NR_CASSCF
    from gnme.utils import orthogonalise

    # Define test H_2 
    mol = gto.M(atom="H 0 0 0; H 2 0 0", 
                basis="6-31g",
                charge=0,
                spin=0)
    myhf = mol.RHF().run()

    # Get AO overlap matrix (metric tensor)
    metric = mol.intor('int1e_ovlp')
    # Define CAS space
    cas = (2,2)

    # Define ground state
    cas1 = NR_CASSCF(myhf,cas[0],cas[1],thresh=1e-7)
    cas1._initMO=orthogonalise(np.array([[0.299266,   0.305182,  0.814001,   0.958884],
                                         [0.465916,   0.475126,  -0.744077,  -0.876515],
                                         [0.305182,  -0.299266,   0.958884,  -0.814001],
                                         [0.475126,  -0.465916,  -0.876515,   0.744077]]), 
                               metric)
    cas1._initCI=orthogonalise(np.array([[  0.705144,   -0.0391702,    0.0446935,     0.706569], 
                                         [ 0.000000,     0.997867,   0.00115663,  -0.00923485],
                                         [-0.000000,  0.000995406,     0.998971,   -0.0262065],
                                         [ -0.705144,    0.0522101,  -0.00762105,     0.707098]]),
                               np.eye(cas1.nDet))
    cas1.kernel()

    cas2 = NR_CASSCF(myhf,cas[0],cas[1],thresh=1e-7)
    cas2._initMO=orthogonalise(np.array([[   0.814001,   0.958884,    0.299266,   0.305182],
                                         [  -0.744077,  -0.876515,    0.465916,   0.475126],
                                         [   0.958884,  -0.814001,    0.305182,  -0.299266],
                                         [  -0.876515,   0.744077,    0.475126,  -0.465916]]),
                               metric)
    cas2._initCI=orthogonalise(np.array([[  0.705144,   -0.0391702,    0.0446935,     0.706569], 
                                         [ 0.0646109,     0.997867,   0.00115663,  -0.00923485],
                                         [-0.0370021,  0.000995406,     0.998971,   -0.0262065],
                                         [ -0.705144,    0.0522101,  -0.00762105,     0.707098]]),
                               np.eye(cas2.nDet))
    cas2.kernel()

    print("Projecting CAS_2 into active space for CAS_1:")
    projvec = cas_proj(cas1,cas2,metric)
    print(projvec)

    print("Total overlap = {:20.10f}".format(cas1.mat_ci[:,0].dot(projvec)))
