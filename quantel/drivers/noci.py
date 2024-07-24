#!/usr/bin/python3

import numpy
from pygnme import utils

eh2ev=27.211386245988

def oscillator_strength(wfnlist, ref_ind=0, plev=1):
    """Compute oscillator strengths from a given reference states [ref_ind]"""
    # Get number of states
    nstate = len(wfnlist)

    print()
    print("===============================================")
    print(" Computing oscillator strengths from solution {:d}".format(ref_ind+1))
    print("===============================================")

    # Get the reference state and integrals
    ref_state = wfnlist[ref_ind]
    ref_state.update_integrals()

    # Loop over the remaining states
    strengths=[]
    for i, state_i in enumerate(wfnlist):
        if(i==ref_ind): continue
        state_i.update_integrals()

        # Compute excitation energy
        de = state_i.energy - ref_state.energy 
        # Compute TDM
        s, tdm = ref_state.tdm(state_i)
        # Compute oscillator strength
        f = 2./3. * de * numpy.dot(tdm,tdm)
        # Convert excitation energy to eV 
        de *= eh2ev

        strengths.append((de, f, s))

    # Print the outcome
    print("{:4s}   {:10s}   {:10s}   {:10s}".format("", "   dE / eV", "   f / au", "   S / au"))
    print("-----------------------------------------------")
    #strengths.sort()
    for i, (de, f, s) in enumerate(strengths):
        print("{: 4d}:  {: 10.6f}   {: 10.6f}   {: 10.6f}".format(i,de,f,s))
    print("----------------------------------------------------------")

    # Record the output in a file
    numpy.savetxt('oscillators', numpy.array(strengths)[:,[0,1]], fmt="% 10.6f")

    return 

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

    # Save to disk for safekeeping
    if save:
        numpy.savetxt('noci_ov',  Swx, fmt="% 8.6f")

    # Print Hamiltonian and Overlap matrices 
    if plev > 0:
        print("\nOverlap Matrix")
        print(Swx)
    print("-----------------------------------------------")

    return Swx

def noci(wfnlist, lindep_tol=1e-8, plev=1):
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
        print(i)
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

    print("\n NOCI Eigenvalues")
    print(w)
    if plev > 0:
        print("\nNOCI Eigenvectors")
        print(v)
    print("\n-----------------------------------------------")

    return Hwx, Swx, eigval, v

