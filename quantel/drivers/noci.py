#!/usr/bin/python3
import numpy
from pygnme import utils
from pyscf.tools import cubegen 
from quantel.gnme.analysis_utils import compute_wfnlist_1rdms, natural_orbitals, get_chergwin_coulson_weights, osc_strength, noci_osc_strength

def overlap(wfnlist, lindep_tol=1e-8, plev=1, save=True):
    """"Perform a NOCI calculation for wavefunctions defined in wfnlist"""

    # Get number of states
    nstate = len(wfnlist)

    print()
    print("-----------------------------------------------")
    print(" Computing nonorthogonal overlap for {:d} solutions".format(nstate))
    print("-----------------------------------------------")

    if plev > 0: print(" > Building NOCI matrices...", end="")
    # Compute Hamiltonian and overlap matrices
    Swx = numpy.zeros((nstate, nstate))
    for i, state_i in enumerate(wfnlist):
        for j, state_j in enumerate(wfnlist):
            if(i<j): continue
            Swx[i,j] = state_i.overlap(state_j)
    if plev > 0: print(" done")

    # Print Hamiltonian and Overlap matrices 
    if plev > 0:
        print("\nOverlap Matrix")
        print(Swx)
    print("-----------------------------------------------")

    return Swx

def noci(wfnlist, natorb_states, lindep_tol=1e-8, plev=1):
    """"Perform a NOCI calculation for wavefunctions defined in wfnlist"""

    #oscillator_strengths(wfnlist, 0)

    # Get number of states
    nstate = len(wfnlist)

    print()
    print("-----------------------------------------------")
    print(" Performing Nonorthogonal CI on {:d} solutions".format(nstate))
    print("-----------------------------------------------")

    if plev > 0: print(" > Building NOCI matrices...", end="")
    # Compute Hamiltonian and overlap matrices
    Hwx = numpy.zeros((nstate, nstate))
    Swx = numpy.zeros((nstate, nstate))
    for i, state_i in enumerate(wfnlist):
        for j, state_j in enumerate(wfnlist):
            if(i<j): continue
            Swx[i,j], Hwx[i,j] = state_i.hamiltonian(state_j)
            Swx[j,i], Hwx[j,i] = Swx[i,j], Hwx[i,j]
    if plev > 0: print(" done")

    # Save to disk for safekeeping
    numpy.savetxt('noci_ov',  Swx, fmt="% 8.6f")
    numpy.savetxt('noci_ham', Hwx, fmt="% 8.6f")

    # Print Hamiltonian and Overlap matrices 
    if plev > 0:
        print("\nNOCI Hamiltonian")
        print(Hwx)
        print("\nNOCI Overlap")
        print(Swx)

    # Solve generalised eigenvalue problem using libgnme
    if plev > 0: print("\n > Solving generalised eigenvalue problem...", end="")
    eigval, v = utils.gen_eig_sym(nstate, Hwx, Swx, thresh=1e-8)
    w = eigval[0,:]
    if plev > 0: print(" done")

    # Save eigenvalues and eigenvectors to disk
    numpy.savetxt('noci_energy_list', w,fmt="% 16.10f")
    numpy.savetxt('noci_evecs', v, fmt="% 8.6f")
    
    # Calc Chergwin-Coulson weights and save overlap 
    ccW = get_chergwin_coulson_weights(Swx, v) 
    numpy.savetxt('noci_weights', ccW, fmt="% 8.6f")
    
    print("\n NOCI Eigenvalues")
    print(w)
    if plev > 0:
        print("\nNOCI Eigenvectors")
        print(v)
        print("\n") 
    ## Calc NOCI WFN oscillator strengths
    #if len(wfnlist) > 1: 
    #    noci_osc_strengths = nociwfn_osc_strength(v, w, wfnlist, noci_ref_ind=0, plev=1)
    #    numpy.savetxt("noci_oscillators", noci_osc_strengths, fmt="% 16.10f")
   
    # Natural Orbitals
    noon_thresh = 0.1 
    state_indices = numpy.arange(v.shape[1])
    noci_rdm1_array = compute_wfnlist_1rdms(wfnlist, v) 
    for state_index in range(natorb_states): 
        noons, norbs = natural_orbitals(noci_rdm1_array, state_index, wfnlist[0].integrals.overlap_matrix() ) 
        for ind, noon in enumerate(noons): 
            if noon > noon_thresh: 
                cubegen.orbital(wfnlist[0].integrals.mol, f"state_{state_index}.norb.{ind}.cube", norbs[:,ind])
 
    nucl_dip, ao_dip = wfnlist[0].integrals.dipole_matrix() 
    metric = wfnlist[0].integrals.overlap_matrix() 
    noci_osc_strengths = noci_osc_strength(noci_rdm1_array, w, ao_dip, metric)
    numpy.savetxt("noci_oscillators", noci_osc_strengths, fmt="% 16.10f")
    ### Then we are going to do what with this? we should compare this against the CSF oscillators ... 

    print("\n-----------------------------------------------")
    return Hwx, Swx, eigval, v
