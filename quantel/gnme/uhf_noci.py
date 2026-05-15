from pygnme import wick, slater, owndata
from quantel.gnme.utils import occstring_to_bitset, generalised_slater_condon
import quantel
import numpy as np

def evaluate_s2(uhf1, uhf2): 
    # Number of orbitals and basis functions
    assert(uhf1.nmo == uhf2.nmo)
    assert(uhf1.nbsf == uhf2.nbsf)
    assert(uhf1.nalfa == uhf2.nalfa) 
    assert(uhf1.nbeta == uhf2.nbeta) 
    nbsf = uhf1.nbsf 
    nmo  = uhf1.nmo     
    metric = owndata(uhf1.integrals.overlap_matrix()) 
    nalfa = uhf1.nalfa 
    nbeta = uhf1.nbeta     

    # Initialise Slater USCF object 
    mb = slater.slater_uscf[float, float, float](nbsf, nmo, nalfa, nbeta, metric)
    
    Cxa = owndata(uhf1.mo_coeff[0][:,:nalfa]) 
    Cxb = owndata(uhf1.mo_coeff[1][:,:nbeta]) 
    
    Cwa = owndata(uhf2.mo_coeff[0][:,:nalfa]) 
    Cwb = owndata(uhf2.mo_coeff[1][:,:nbeta])

    
    # Evaluate S2 expectation value  
    S2 = 0.0 
    S2 = mb.evaluate_s2(Cxa, Cxb, Cwa, Cwb, S2)
    return S2 


def uhf_occstring(uhf):
    occstr = ""
    for _ in range(min(uhf.nalfa, uhf.nbeta)):
        occstr += "2" 
    if uhf.nalfa > uhf.nbeta:
        for _ in range(uhf.nalfa - uhf.nbeta):  
            occstr += "a"
    elif uhf.nalfa < uhf.nbeta:      
        for _ in range(uhf.nbeta - uhf.nalfa):  
            occstr += "b"
    return occstr 

def uhf_rdm1(uhf1, uhf2, metric, thresh=1e-10, enuc = 0.0 ):
    # Convert integral matrices to pygnme-friendly format
    ovlp = owndata(metric)

    # Number of orbitals and basis functions
    assert(uhf1.nmo == uhf2.nmo)
    assert(uhf1.nbsf == uhf2.nbsf)
    nmo = uhf1.nmo
    nbsf = uhf1.nbsf

    # Initialize output
    Hxw, Sxw = 0, 0

    # Setup biorthogonalised orbital pair
    c1a = uhf1.mo_coeff[0].copy()
    c2a = uhf2.mo_coeff[0].copy()
    c1b = uhf1.mo_coeff[1].copy()
    c2b = uhf2.mo_coeff[1].copy()
    ##   
    uhf1_nopen = np.abs(uhf1.nalfa - uhf1.nbeta) 
    uhf2_nopen = np.abs(uhf2.nalfa - uhf2.nbeta)
    uhf1_ncore = min(uhf1.nalfa, uhf1.nbeta) 
    uhf2_ncore = min(uhf2.nalfa, uhf2.nbeta) 
    ##
    refxa = wick.reference_state[float](nbsf,nmo,uhf1.nalfa,uhf1_nopen,uhf1_ncore,owndata(c1a))
    refxb = wick.reference_state[float](nbsf,nmo,uhf1.nbeta,uhf1_nopen,uhf1_ncore,owndata(c1b))
    refwa = wick.reference_state[float](nbsf,nmo,uhf2.nalfa,uhf2_nopen,uhf2_ncore,owndata(c2a))
    refwb = wick.reference_state[float](nbsf,nmo,uhf2.nbeta,uhf2_nopen,uhf2_ncore,owndata(c2b))

    # Setup paired orbitals
    orba = wick.wick_orbitals[float, float](refxa, refwa, ovlp)
    orbb = wick.wick_orbitals[float, float](refxb, refwb, ovlp)

    # Setup matrix builder object
    mb = wick.wick_uscf[float, float, float](orba, orbb, enuc)

    # Construct detx and detw for uhf determinants
    detx = uhf_occstring(uhf1) 
    detw = uhf_occstring(uhf2) 

    # Evaluate coupling term 
    bax, bbx = occstring_to_bitset(detx)
    baw, bbw = occstring_to_bitset(detw)
    Pa, Pb = (np.zeros((nmo,nmo), dtype=float), np.zeros((nmo,nmo), dtype=float)) 
    Sxw = mb.evaluate_rdm1(bax, bbx, baw, bbw, 0.0, Pa, Pb)
    return Sxw, Pa.T, Pb.T 

def uhf_coupling(uhf1, uhf2, metric, hcore=None, eri=None, enuc=0.0, thresh=1e-10):
    # Convert integral matrices to pygnme-friendly format
    ovlp = owndata(metric)

    # Number of orbitals and basis functions
    assert(uhf1.nmo == uhf2.nmo)
    assert(uhf1.nbsf == uhf2.nbsf)
    nmo = uhf1.nmo
    nbsf = uhf1.nbsf

    # Setup biorthogonalised orbital pair
    c1a = uhf1.mo_coeff[0].copy()
    c2a = uhf2.mo_coeff[0].copy()
    c1b = uhf1.mo_coeff[1].copy()
    c2b = uhf2.mo_coeff[1].copy()
    ##   
    uhf1_nopen = np.abs(uhf1.nalfa - uhf1.nbeta) 
    uhf2_nopen = np.abs(uhf2.nalfa - uhf2.nbeta)
    uhf1_ncore = min(uhf1.nalfa, uhf1.nbeta) 
    uhf2_ncore = min(uhf2.nalfa, uhf2.nbeta) 
    ##
    refxa = wick.reference_state[float](nbsf,nmo,uhf1.nalfa,uhf1_nopen,uhf1_ncore,owndata(c1a))
    refxb = wick.reference_state[float](nbsf,nmo,uhf1.nbeta,uhf1_nopen,uhf1_ncore,owndata(c1b))
    refwa = wick.reference_state[float](nbsf,nmo,uhf2.nalfa,uhf2_nopen,uhf2_ncore,owndata(c2a))
    refwb = wick.reference_state[float](nbsf,nmo,uhf2.nbeta,uhf2_nopen,uhf2_ncore,owndata(c2b))

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
    
    # Construct detx and detw for uhf determinants
    detx = uhf_occstring(uhf1) 
    detw = uhf_occstring(uhf2) 

    # Evaluate coupling term 
    bax, bbx = occstring_to_bitset(detx)
    baw, bbw = occstring_to_bitset(detw)
    Sxw, Hxw = mb.evaluate(bax, bbx, baw, bbw)
    
    return Sxw, Hxw
