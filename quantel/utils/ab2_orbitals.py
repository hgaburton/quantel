from quantel.utils.linalg import orthogonalise 
from quantel.utils.orbital_utils import compute_mulliken_atomic_pops, localise_orbitals
import numpy as np

def get_ab2_orbs(wfn):
    """
    Implementation of method to construct 1:1 Antibonding orbitals using MP2 amplitudes as proposed in (Aldossary, 2022)
    """
    #NOTE: only implemented for the RHF wave function  
    # Localise and select occupied orbitals 
    lumo_idx = wfn.nocc
    wfn.localise_orbitals()
    ds, bond_indices = compute_mulliken_atomic_pops(wfn.integrals.mol, wfn.mo_coeff[:,:lumo_idx].copy(), give_bond_indices=True)
    
    # Store Fock matrix in new MO basis 
    ref_mo = wfn.mo_coeff.copy()
    fock_mo = np.linalg.multi_dot((ref_mo.T, wfn.fock, ref_mo))
    
    # Compute AB2 orbitals 
    AB2_orbitals = np.zeros((wfn.nbsf,len(bond_indices)))
    for idx,BO in enumerate(bond_indices):
        # Construct the MO exchange matrix with localised occupied orbitals  
        dens = np.outer(ref_mo[:,BO],ref_mo[:,BO].T)
        
        ao_exchange = np.einsum('pqrs,qs->pr', wfn.integrals.tei_array(True, False) , dens, optimize="optimal")
        vir_mo_exchange = np.linalg.multi_dot((ref_mo[:,lumo_idx:].T, ao_exchange, ref_mo[:,lumo_idx:]))
        
        # Scale with MP2 denominator
        scaling = True
        neg_defin = True 
        for va in range(lumo_idx, wfn.nmo): 
            for vb in range(lumo_idx, wfn.nmo):
                delta = - fock_mo[va,va] - fock_mo[vb,vb] + 2*fock_mo[BO,BO]
                vir_mo_exchange[va - lumo_idx, vb - lumo_idx] /= delta
                if delta > 0: 
                    scaling = False 
        
        # Compute and store lowest eigenvector  
        eigs, eigvecs = np.linalg.eigh(vir_mo_exchange)
        if np.any(eigs >0): 
            neg_defin = False 

        # Select MO and Transform back into AO basis
        orbital = eigvecs[:,np.argmin(eigs)] 
        AB2_orbitals[:,idx] = np.ndarray.flatten(np.dot(ref_mo[:,lumo_idx:], orbital))
   
    return AB2_orbitals

def reorthogonalise_virs(wfn, additional_virs):
    # Reconstruct virtual MO basis
    vir = np.c_[additional_virs, wfn.mo_coeff[:,wfn.nocc:].copy()]
    # Remove linear dependencies and re-orthogonalise
    ortho_virs = orthogonalise(vir, wfn.integrals.overlap_matrix(), fill=False, modified=True) 
     
    # Return orthogonalise virs with correct shape
    if ortho_virs.shape[1]==(wfn.nmo-wfn.nocc):
        return ortho_virs 
    else:
        print(f"Warning: GS yielded incorrect number of linearly dependent vectors, result {ortho_virs.shape[1]} vecs -> desired {wfn.nmo-wfn.nocc}")
        return ortho_virs[:,:wfn.nmo-wfn.nocc] 


def get_smushed_ab2_orbs(wfn, indices):
    """
    Computing more than one AB2 orbital at a time
        
    Inputs
        - indices: indices of the bonding orbitals wish to "smush" 
          together for AB2 calc 
    """
    #NOTE: not extensively tested 
    # Localise and select occupied orbitals 
    lumo_idx = wfn.nocc
    wfn.localise_orbitals()
    ds, bond_indices = compute_mulliken_atomic_pops(wfn.integrals.mol, wfn.mo_coeff[:,:lumo_idx].copy(), give_bond_indices=True)
    
    # Store Fock matrix in new MO basis 
    ref_mo = wfn.mo_coeff.copy()
    fock_mo = np.linalg.multi_dot((ref_mo.T, wfn.fock, ref_mo))
    
    # Compute AB2 orbitals 
    AB2_orbitals = np.zeros((wfn.nbsf,len(bond_indices)))
    
    # Construct the MO exchange matrix with localised occupied orbitals  
    exchange = np.zeros((wfn.nmo-wfn.nocc, wfn.nmo-wfn.nocc), dtype=float)
    for idx in bond_indices[indices]:
        dens = np.outer(ref_mo[:,idx],ref_mo[:,idx].T)
        ao_exchange = np.einsum('pqrs,qs->pr', wfn.integrals.tei_array(True, False) , dens, optimize="optimal")
        vir_mo_exchange = np.linalg.multi_dot((wfn.mo_coeff[:,lumo_idx:].T, ao_exchange, wfn.mo_coeff[:,lumo_idx:]))
    
        # Scale with MP2 denominator
        scaling = True
        neg_defin = True 
        for va in range(lumo_idx, wfn.nmo): 
            for vb in range(lumo_idx, wfn.nmo):
                delta = - fock_mo[va,va] - fock_mo[vb,vb] + 2*fock_mo[idx,idx]
                vir_mo_exchange[va - wfn.nocc, vb - wfn.nocc] /= delta
                if delta > 0: 
                    scaling = False
        exchange += vir_mo_exchange 
    
    # Compute and store lowest eigenvector  
    eigs, eigvecs = np.linalg.eigh(exchange)
    if np.any(eigs > 0): 
        neg_defin = False 

    # Select MO and Transform back into AO basis
    for i in range(len(indices)):
        AB2_orbitals[:,i] = (wfn.mo_coeff[:,lumo_idx:] @ eigvecs[:,i]).flatten() 

    #orb2 = eigvecs[:,1]
    #ab2_orbitals[:,1] = (wfn.mo_coeff[:,lumo_idx:] @ orb2).flatten() 
    
    # Localise orbitals
    coeffs, isstable = localise_orbitals(wfn.integrals.mol, AB2_orbitals) 
    return coeffs
