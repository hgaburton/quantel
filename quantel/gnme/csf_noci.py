from pygnme import wick, owndata
from quantel.gnme.utils import occstring_to_bitset
from quantel.utils.csf_utils import get_csf_vector

def csf_coupling(csf1, csf2, metric, hcore=None, eri=None, enuc=0.0, thresh=1e-10):
    # Convert integral matrices to pygnme-friendly format
    ovlp = owndata(metric)

    # Number of orbitals and basis functions
    assert(csf1.nmo == csf2.nmo)
    assert(csf1.nbsf == csf2.nbsf)
    nmo = csf1.nmo
    nbsf = csf1.nbsf

    # Initialize output
    Hxw, Sxw = 0, 0

    # Setup biorthogonalised orbital pair
    c1 = csf1.mo_coeff.copy()
    c2 = csf2.mo_coeff.copy()
    refxa = wick.reference_state[float](nbsf,nmo,csf1.nalfa,csf1.cas_nmo,csf1.ncore,owndata(c1))
    refxb = wick.reference_state[float](nbsf,nmo,csf1.nbeta,csf1.cas_nmo,csf1.ncore,owndata(c1))
    refwa = wick.reference_state[float](nbsf,nmo,csf2.nalfa,csf2.cas_nmo,csf2.ncore,owndata(c2))
    refwb = wick.reference_state[float](nbsf,nmo,csf2.nbeta,csf2.cas_nmo,csf2.ncore,owndata(c2))

    # Setup paired orbitals
    orba = wick.wick_orbitals[float, float](refxa, refwa, ovlp)
    orbb = wick.wick_orbitals[float, float](refxb, refwb, ovlp)

    # Setup matrix builder object
    mb = wick.wick_uscf[float, float, float](orba, orbb, enuc)

    # Add one- and two-body contributions
    if(hcore is not None):
        h1e = owndata(hcore)
        mb.add_one_body(h1e)
    if(eri is not None):
        h2e = owndata(eri)
        mb.add_two_body(h2e)

    for (detx, cix) in get_csf_vector(csf1.spin_coupling):
        if(abs(cix) < thresh):
            continue
        bax, bbx = occstring_to_bitset(detx)
        
        for detw, ciw in get_csf_vector(csf2.spin_coupling):
            if(abs(ciw) < thresh):
                continue
            baw, bbw = occstring_to_bitset(detw)

            stmp, htmp = mb.evaluate(bax, bbx, baw, bbw)
            Hxw += htmp * cix * ciw
            Sxw += stmp * cix * ciw

    return Sxw, Hxw
