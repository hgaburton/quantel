from pygnme import wick, owndata
from quantel.gnme.utils import occstring_to_bitset

def csf_coupling(csf1, csf2, metric, hcore=None, eri=None, enuc=0.0, thresh=1e-10):
    # Convert integral matrices to pygnme-friendly format
    ovlp = owndata(metric)

    # Number of orbitals
    assert(csf1.nmo == csf2.nmo)
    nmo = csf1.nmo
    # Number of electrons
    assert(csf1.nalfa == csf2.nalfa)
    assert(csf1.nbeta == csf2.nbeta)
    # For now, we can only do ms = 0
    assert(csf1.nalfa == csf1.nbeta)
    nocc = csf1.nalfa
    
    # Initialize output
    Hxw, Sxw = 0, 0

    # Setup biorthogonalised orbital pair
    refx = wick.reference_state[float](nmo,nmo,nocc,csf1.cas_nmo,csf1.ncore,owndata(csf1.mo_coeff))
    refw = wick.reference_state[float](nmo,nmo,nocc,csf2.cas_nmo,csf2.ncore,owndata(csf2.mo_coeff))

    # Setup paired orbitals
    orbs = wick.wick_orbitals[float, float](refx, refw, ovlp)

    # Setup matrix builder object
    mb = wick.wick_rscf[float, float, float](orbs, enuc)

    # Add one- and two-body contributions
    if(hcore is not None):
        h1e = owndata(hcore)
        mb.add_one_body(h1e)
    if(eri is not None):
        h2e = owndata(eri)
        mb.add_two_body(h2e)

    for cix, detx in zip(csf1.civec, csf1.detlist):
        if(abs(cix) < thresh):
            continue
        bax, bbx = occstring_to_bitset(detx)
        
        for ciw, detw in zip(csf2.civec, csf2.detlist):
            if(abs(ciw) < thresh):
                continue
            baw, bbw = occstring_to_bitset(detw)
            
            stmp, htmp = mb.evaluate(bax, bbx, baw, bbw)
            Hxw += htmp * cix * ciw
            Sxw += stmp * cix * ciw

    return Sxw, Hxw