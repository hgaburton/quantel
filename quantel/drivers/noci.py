#!/usr/bin/python3
import numpy
from pygnme import utils
from quantel.gnme.utils import gen_eig_sym
from quantel.gnme.utils import get_chergwin_coulson_weights

eh2ev=27.211386245988

def oscillator_strength(wfnlist, namelist, ref_ind=0, plev=1):
    """Compute oscillator strengths from a given reference states [ref_ind]"""
    # Get number of states
    nstate = len(wfnlist)

    print()
    print("===============================================")
    print(" Computing oscillator strengths from solution {:d}".format(ref_ind+1))
    print("===============================================")

    # Get the reference state and integrals
    ref_state = wfnlist[ref_ind]
    ref_state.update()

    # Loop over the remaining states
    strengths=[]
    for i, state_i in enumerate(wfnlist):
        if(i==ref_ind): continue
        state_i.update()

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
    print("{:4s}   {:10s}   {:10s}   {:10s}".format("","  dE / eV", "   f / au", "   S / au"))
    print("-----------------------------------------------")
    #strengths.sort()
    for i, (de, f, s) in enumerate(strengths):
        print("{}:  {: 10.6f}   {: 10.6f}   {: 10.6f}".format(namelist[i+1],de,f,s))
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
    eigval, v = gen_eig_sym(Hwx, Swx, thresh=1e-8)
    w = eigval[0,:]
    if plev > 0: print(" done")

    # Save eigenvalues and eigenvectors to disk
    numpy.savetxt('noci_energy_list', w,fmt="% 16.10f")
    numpy.savetxt('noci_evecs', v, fmt="% 8.6f")
    
    #Get Chergwin-Coulson weights and save overlap 
    ccW = get_chergwin_coulson_weights(Swx, v) 
    numpy.savetxt('noci_weights', ccW, fmt="% 8.6f")

    print("\n NOCI Eigenvalues")
    print(w)
    if plev > 0:
        print("\nNOCI Eigenvectors")
        print(v)
    print("\n-----------------------------------------------")

    return Hwx, Swx, eigval, v

def selected_noci(wfnlist, lindep_tol=1e-8, plev=1):
    """Selected NOCI: iteratively build the P space by adding states from Q
    one at a time, choosing the state with the largest perturbative correction.

    For each Q state q the correction is:
        W_q = [Cp @ (Hpq - E0*Spq)]^2 / (Hqq - E0*Sqq)
    where Cp is the ground-state eigenvector in the current P space,
    E0 is the corresponding eigenvalue, and Hqq/Sqq are assumed diagonal.
    """
    nstate = len(wfnlist)

    print()
    print("-----------------------------------------------")
    print(" Performing selected Nonorthogonal CI on {:d} solutions".format(nstate))
    print("-----------------------------------------------")

    # Sort states by diagonal energy so P starts with the best reference
    energies = numpy.array([state.energy for state in wfnlist])
    order = numpy.argsort(energies)
    wfnlist = [wfnlist[i] for i in order]

    # Full H and S matrices, filled lazily
    Hwx = numpy.zeros((nstate, nstate))
    Swx = numpy.zeros((nstate, nstate))
    computed = numpy.zeros((nstate, nstate), dtype=bool)

    def ensure_row(i):
        """Compute H/S matrix elements for row i against every state."""
        for j in range(nstate):
            if not computed[i, j]:
                Swx[i, j], Hwx[i, j] = wfnlist[i].hamiltonian(wfnlist[j])
                Swx[j, i], Hwx[j, i] = Swx[i, j], Hwx[i, j]
                computed[i, j] = computed[j, i] = True

    # Compute all diagonal elements upfront (needed for Q-space denominators)
    for i in range(nstate):
        if not computed[i, i]:
            Swx[i, i], Hwx[i, i] = wfnlist[i].hamiltonian(wfnlist[i])
            computed[i, i] = True

    # P space: start with the lowest-energy state
    ensure_row(0)
    p_indices = [0]
    q_indices = list(range(1, nstate))

    if plev > 0:
        print(f"\n  {'Step':>4s}  {'Added':>6s}  {'E0 / Eh':>16s}  {'W_best':>12s}  {'E_added / Eh':>16s}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*16}  {'─'*12}  {'─'*16}")

    eps = 1e-6
    E_pert_total = 0.0

    if plev > 0:
        print(f"\n  {'|P|':>4s}  {'Added':>6s}  {'E0 / Eh':>16s}  {'W_best':>12s}  {'E_pert_sum':>14s}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*16}  {'─'*12}  {'─'*14}")

    while q_indices:
        p = numpy.array(p_indices)
        q = numpy.array(q_indices)

        # Symmetrize P-space submatrices to avoid numerical drift warnings
        Hpp = Hwx[numpy.ix_(p, p)]
        Spp = Swx[numpy.ix_(p, p)]
        Hpp = 0.5 * (Hpp + Hpp.T)
        Spp = 0.5 * (Spp + Spp.T)

        # Solve the generalised eigenvalue problem in the current P space
        eigval, eigvec = gen_eig_sym(Hpp, Spp, thresh=lindep_tol)
        E0 = float(numpy.ravel(eigval)[0])
        Cp = numpy.ravel(eigvec[:, 0])   # ground-state coefficients in P basis (|P|,)

        # Coupling between P and Q spaces
        Hpq = Hwx[numpy.ix_(p, q)]    # (|P|, |Q|)
        Spq = Swx[numpy.ix_(p, q)]    # (|P|, |Q|)
        # Diagonal Q elements (Hqq and Sqq assumed diagonal)
        Hqq_diag = numpy.diag(Hwx)[q]
        Sqq_diag = numpy.diag(Swx)[q]

        # Remove Q states whose overlap with the current NOCI wavefunction is ~1
        overlap_with_p = Cp.T @ Spq                  # (|Q|,)
        redundant = numpy.abs(1.0 - overlap_with_p) < 1e-6
        if numpy.any(redundant):
            removed = [q_indices[k] for k in numpy.where(redundant)[0]]
            if plev > 0:
                print(f"  Removing {len(removed)} Q state(s) with overlap ~1: {removed}")
            q_indices = [qi for qi, r in zip(q_indices, redundant) if not r]
            if not q_indices:
                break
            q = numpy.array(q_indices)
            Hpq = Hpq[:, ~redundant]
            Spq = Spq[:, ~redundant]
            Hqq_diag = Hqq_diag[~redundant]
            Sqq_diag = Sqq_diag[~redundant]

        # Perturbative correction: W_q = -(Cp @ (Hpq - E0*Spq))^2 / (Hqq - E0*Sqq)
        coupling = Cp @ (Hpq - E0 * Spq)        # (|Q|,)
        denom = Hqq_diag - E0 * Sqq_diag        # (|Q|,)
        W = - coupling ** 2 / denom              # (|Q|,)

        # Select Q state with the largest |W|
        best_local = int(numpy.argmax(numpy.abs(W)))
        best_W = W[best_local]

        # Stop if the best correction is below epsilon
        if abs(best_W) < eps:
            if plev > 0:
                print(f"  Best perturbative correction {best_W:.4e} < eps={eps:.1e}; stopping.")
            break

        best_global = q_indices[best_local]
        Ept2 = numpy.sum(W)

        if plev > 0:
            print(f"  {len(p_indices):4d}  {best_global:6d}  {E0: 16.10f}  "
                  f"{best_W: 12.4e}  {Ept2: 14.6e}")

        # Move best Q state into P and compute its full row
        p_indices.append(best_global)
        q_indices.remove(best_global)
        ensure_row(best_global)

    # Final NOCI using only the selected P-space states
    p = numpy.array(p_indices)
    Hpp_final = Hwx[numpy.ix_(p, p)]
    Spp_final = Swx[numpy.ix_(p, p)]
    Hpp_final = 0.5 * (Hpp_final + Hpp_final.T)
    Spp_final = 0.5 * (Spp_final + Spp_final.T)

    eigval_final, v_final = gen_eig_sym(Hpp_final, Spp_final, thresh=lindep_tol)
    w = numpy.ravel(eigval_final)

    # Save results to disk
    numpy.savetxt('noci_ov',  Spp_final, fmt="% 8.6f")
    numpy.savetxt('noci_ham', Hpp_final, fmt="% 8.6f")
    numpy.savetxt('noci_energy_list', w, fmt="% 16.10f")
    numpy.savetxt('noci_evecs', v_final, fmt="% 8.6f")

    ccW = get_chergwin_coulson_weights(Spp_final, v_final[:,0])
    numpy.savetxt('noci_weights', ccW, fmt="% 8.6f")

    if plev > 0:
        print(f"\n  Selected NOCI: {len(p_indices)} states in P space")
        print(f"  Variational energy   : {w[0]: 16.10f} Eh")
        print(f"  Perturbative corr.   : {Ept2: 16.10f} Eh")
        print(f"  Total (var + pert)   : {w[0] + Ept2: 16.10f} Eh")
        print(f"  All eigenvalues: {w}")
    print("-----------------------------------------------")

    return Hpp_final, Spp_final, eigval_final, v_final
