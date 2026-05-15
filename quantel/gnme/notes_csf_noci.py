from pygnme import wick, owndata
from quantel.gnme.utils import occstring_to_bitset, generalised_slater_condon
from quantel.utils.csf_utils import get_csf_vector
import quantel
import numpy as np

def csf_coupling(csf1, csf2, metric, hcore=None, eri=None, enuc=0.0, thresh=1e-10):
    # Convert integral matrices to pygnme-friendly format
    ovlp = owndata(metric)

    # Number of orbitals and basis functions
    assert(csf1.nmo == csf2.nmo)
    assert(csf1.nbsf == csf2.nbsf)
    nmo = csf1.nmo
    nbsf = csf1.nbsf

    # Initialize output
    # for the diple we would only care about the TDM output 
    Hxw, Sxw = 0, 0

    # Setup biorthogonalised orbital pair
    c1 = csf1.mo_coeff.copy()
    c2 = csf2.mo_coeff.copy()
    refxa = wick.reference_state[float](nbsf,nmo,csf1.nalfa,csf1.nopen,csf1.ncore,owndata(c1))
    refxb = wick.reference_state[float](nbsf,nmo,csf1.nbeta,csf1.nopen,csf1.ncore,owndata(c1))
    refwa = wick.reference_state[float](nbsf,nmo,csf2.nalfa,csf2.nopen,csf2.ncore,owndata(c2))
    refwb = wick.reference_state[float](nbsf,nmo,csf2.nbeta,csf2.nopen,csf2.ncore,owndata(c2))

    # Setup paired orbitals
    orba = wick.wick_orbitals[float, float](refxa, refwa, ovlp)
    orbb = wick.wick_orbitals[float, float](refxb, refwb, ovlp)

    # Setup matrix builder object
    mb = wick.wick_uscf[float, float, float](orba, orbb, enuc)
    # would this just be the nuclear dipole as the scalar component 
    #mb = wick.wick_uscf[float, float, float](orba, orbb, nuc_dip)

    # Add one- and two-body contributions
    if(hcore is not None):
        h1e = owndata(hcore)
        mb.add_one_body(h1e)
    if(eri is not None):
        h2e = owndata(eri)
        mb.add_two_body(h2e)
    # there is only a one body component for the TDM right so would be something like this: 
    #if(ao_dip is not None):
    #    ao_dip1e = owndata(ao_dip)
    #    mb.add_one_body(ao_dip1e)

    # this converts the csf vector - i.e determinant expansion
    for (detx, cix) in get_csf_vector(csf1.spin_coupling):
        if(abs(cix) < thresh): #if coefficient is above a threshold
            continue
        # this is the bitset representation of the determinants. (what is the big diference here - arent they all in binary representation?) 
        bax, bbx = occstring_to_bitset(detx)
        
        for detw, ciw in get_csf_vector(csf2.spin_coupling):
            if(abs(ciw) < thresh):
                continue
            baw, bbw = occstring_to_bitset(detw)

            stmp, htmp = mb.evaluate(bax, bbx, baw, bbw) # ok are these then the overlap and the Hamiltonian coupling terms between the determinants

            # transforming them again
            Hxw += htmp * cix * ciw
            Sxw += stmp * cix * ciw
    # these are two scalars which then are the couplings between the two csfs (sum of all of the couplings between all of the determinants in the CSF)  
    # and Sxw is the overlap 
    return Sxw, Hxw

def csf_coupling_slater_condon(csf1, csf2, ints, thresh=1e-10):
    # Compute coupling of two CSFs using Slater-Condon rules

    # Get the overlap matrix
    metric = ints.overlap_matrix()

    # Initialize output
    Hxw, Sxw = 0, 0

    for (detx, cix) in get_csf_vector(csf1.spin_coupling):
        if(abs(cix) < thresh):
            continue
        # Get occupied orbitals for csf1 determinants
        bax = np.argwhere(np.asarray(list(detx)) == 'a').ravel() + csf1.ncore
        bbx = np.argwhere(np.asarray(list(detx)) == 'b').ravel() + csf1.ncore
        Cax = np.hstack([csf1.mo_coeff[:,:csf1.ncore], csf1.mo_coeff[:,bax]]).copy()
        Cbx = np.hstack([csf1.mo_coeff[:,:csf1.ncore], csf1.mo_coeff[:,bbx]]).copy()

        for (detw, ciw) in get_csf_vector(csf2.spin_coupling):
            if(abs(ciw) < thresh):
                continue
            # Get occupied orbitals for csf2 determinants
            baw = np.argwhere(np.asarray(list(detw)) == 'a').ravel() + csf2.ncore
            bbw = np.argwhere(np.asarray(list(detw)) == 'b').ravel() + csf2.ncore
            Caw = np.hstack([csf2.mo_coeff[:,:csf2.ncore], csf2.mo_coeff[:,baw]]).copy()
            Cbw = np.hstack([csf2.mo_coeff[:,:csf2.ncore], csf2.mo_coeff[:,bbw]]).copy()

            # Compute the matrix elements from generalized Slater-Condon rules
            S, H = generalised_slater_condon(Cax,Cbx,Caw,Cbw,ints)

            # Increment the total Hamiltonian and overlap coupling
            Hxw += H * cix * ciw
            Sxw += S * cix * ciw

    return Sxw, Hxw
