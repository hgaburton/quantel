#!/usr/bin/python3

import numpy
from pygnme import utils

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

    print("\n NOCI Eigenvalues")
    print(w)
    if plev > 0:
        print("\nNOCI Eigenvectors")
        print(v)
    print("\n-----------------------------------------------")

    return Hwx, Swx, eigval, v
