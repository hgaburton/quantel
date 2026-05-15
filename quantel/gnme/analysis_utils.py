#!/usr/bin/python3 
import numpy as np
import pygnme
from quantel.gnme.csf_noci import csf_rdm1 
from pyscf.tools import cubegen 
import scipy 

eh2ev = 27.2114 
# this has no state dependence - so thats fine .. 
def wfnlist_osc_strength(wfnlist, ref_ind=0, plev=1, save=True):
    """Compute oscillator strengths from a given reference states [ref_ind]"""
    print()
    print("===============================================")
    print(" Computing oscillator strengths from solution {:d}".format(ref_ind+1))
    print("===============================================")

    nstate = len(wfnlist)
    # Ref_ind indicates energetic ordering 
    energies = [ wfn.energy for wfn in wfnlist ] 
    inds = np.argsort(energies) 
    ref_ind = inds[ref_ind] 
    ref_state = wfnlist[ref_ind]
    ref_state.update()

    # Loop over the remaining states
    data=[]
    tdms = [] 
    for i, state_i in enumerate(wfnlist):
        if(i==ref_ind): continue
        state_i.update()
        # Compute excitation energy in eV 
        s = ref_state.overlap(state_i)
        de = state_i.energy - ref_state.energy 
        # Compute TDM
        tdm = ref_state.tdm(state_i) 
        # Compute oscillator strength
        f = 2./3. * de * np.dot(tdm,tdm)
        # Convert excitation energy to eV 
        de *= eh2ev
        data.append((de, f, s))
        tdms.append(np.dot(tdm,tdm))

    # Print the outcome
    print("{:4s}   {:10s}   {:10s}   {:10s}".format("","  dE / eV", "   f / au","   S / au"))
    print("-----------------------------------------------")
    #strengths.sort()
    namelist = np.genfromtxt("name_list", dtype=str) 
    for i, (de, f, s) in enumerate(data):
        print("{:2}:  {: 10.6f}   {: 10.6f}   {: 10.6f}".format(namelist[i],de,f,s))
    print("----------------------------------------------------------")
    return data, np.array(tdms) 

def natural_orbitals(noci_rdm1_array, state_index, metric, plev=1): 
    RDM1 = noci_rdm1_array[state_index, state_index,:,:]   
    # contravariant AO  to covariant AO basis  
    RDM1 = np.einsum("kl, km, ln -> mn", RDM1, metric, metric, optimize="optimal") 
    noons, norbs  = scipy.linalg.eigh(a=RDM1,b=metric) 
    inds = np.argsort(noons)[::-1]
    noons = noons[inds]
    norbs = norbs[:, inds] 
    if plev>0: 
        print(f"State {state_index} Natural orbital occupation numbers") 
        print(noons[:20])
    return noons, norbs  

def compute_noci_1rdms(wfnlist, noci_evecs): 
    nmo = wfnlist[0].nmo 
    wfn_rdm1_array = np.zeros( (len(wfnlist), len(wfnlist), nmo, nmo), dtype=float)
    overlaps = np.zeros((len(wfnlist), len(wfnlist)), dtype=float) 
    for indi in range(len(wfnlist)): 
        for indj in range(len(wfnlist)):
            # RDM1 in contravariant AO basis    
            overlaps[indi,indj], rdm1_ij =  wfnlist[indi].trans_rdm1(wfnlist[indj])
            wfn_rdm1_array[indi,indj,:,:] = rdm1_ij

    # Construct density matrices for NOCI states 
    # Swap required: 12D_pq = < 2 | E_pq | 1 > = sum_ij (ci.conj) * cj <i | E_pq | j > = sum_ij (ci.conj) * cj * jiD_pq    
    noci_rdm1_array = np.einsum("jimn, ik, jl -> klmn", wfn_rdm1_array, noci_evecs, noci_evecs, optimize="optimal") 
    # Return WFNs and NOCI densities in Contravariant AO basis 
    return wfn_rdm1_array, noci_rdm1_array  

def noci_osc_strength(noci_rdm1_array, noci_energies, ao_dip, ref_ind=0, plev=1):
    if plev >0: 
        print("=============================") 
        print("  NOCI oscillator strengths") 
        print("=============================") 
        print("{:1s}   {:10s}   {:10s}   {:10s}".format("","  dE / eV", "   f / au", "  TDM / au" ))
    
    F = [] 
    for state in range(1, noci_rdm1_array.shape[1]):
        tdm = np.zeros((3), dtype=float) 
        for x in range(3):
            # NOCI RDM1 given in Contravariant AO basis 
            tdm[x] = np.einsum("ij, ji", ao_dip[x], noci_rdm1_array[ref_ind, state,:,:])
        de = noci_energies[state] - noci_energies[ref_ind]  
        tdm2 = np.dot(tdm,tdm) 
        F.append( 2./3. * de * tdm2 )
        if plev >0:
            print("{}:  {: 10.6f}   {: 10.6f}   {: 10.6f}".format(state,eh2ev*de,F[-1],tdm2))
    return F  

def get_chergwin_coulson_weights(overlap_matrix, evecs): 
    """Construct the Chergwin-Coulson weights for a given overlap matrix and eigenvector of the generalised eigenvalue problem
    
       Inputs:
       -------
           overlap_matrix  2d-array containing the overlap matrix for the nonorthogonal states
           evecs           1d-array containing the eigenvector of the generalised eigenvalue problem
        Outputs:
        --------
            W              1d-array containing the Chergwin-Coulson weights
    """
    return evecs * (overlap_matrix @ evecs)
