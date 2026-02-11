import numpy as np
import scipy 
from pyscf.tools import cubegen 
from pyscf.lo import ibo, iao, pipek
from pyscf.scf import hf
from pyscf.mp import mp2 
import sys
from quantel.utils.linalg import orthogonalise, matrix_print 

def localise_orbs(wfn,indices):    
    """
       Wrapper for PYSCF orbital localisation via Pipek-Mezey scheme
        Input: 
            - wfn object (change to just what we need or update the wfn)
            - lumo_idx : LUMO idx varies on wavefunction ansatz
        Output: 
            - ds : array atomic mulliken populations 
            (on site changes update the wfn object with localised MO coefficients) 
    """
    # Localise occupied orbitals
    pm = pipek.PM(wfn.integrals.mol, wfn.mo_coeff[:,indices].copy())
    pm.pop_method = "mulliken" 
    _ = pm.kernel()
    
    # Perform Stability analysis 
    isstable, local_coeffs  = pm.stability_jacobi()
    if not isstable:
        _ = pm.kernel(local_coeffs)
        isstable, local_coeffs  = pm.stability_jacobi()
        assert( isstable ) 
    
    # Compute Mulliken atomic populations 
    ds =[]
    for idx in range(len(indices)): 
        pop, chg = hf.mulliken_pop(mol=wfn.integrals.mol, dm=np.outer(local_coeffs[:,idx], local_coeffs[:,idx]), verbose=0)
        # Compute mulliken atomic populations 
        chg += - wfn.integrals.mol.atom_charges()
        ds.append(1/np.sum(chg**2))
   
    wfn.mo_coeff[:,indices] = local_coeffs
    wfn.update()  

    return isstable, np.array(ds) 

def get_ab2_orbs(wfn, lumo_idx):
    """
    Implementation of method to construct 1:1 Antibonding orbitals as proposed in (Aldossary, 2022)
    """
    # Localise and select occupied orbitals 
    isstable, ds = wfn.localise()
    bond_indices = np.argwhere(ds>1.5).flatten()
    
    # Store Fock matrix in new MO basis 
    ref_mo = wfn.mo_coeff.copy()
    fock_mo = np.linalg.multi_dot((ref_mo.T, wfn.fock, ref_mo))
    
    # Compute AB2 orbitals 
    ab2_orbs = np.zeros((wfn.nbsf,len(bond_indices)))
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
        eigs, eigvecs = scipy.linalg.eigh(vir_mo_exchange)
        if np.any(eigs >0): 
            neg_defin = False 

        # Select MO and Transform back into AO basis
        orbital = eigvecs[:,np.argmin(eigs)] 
        ab2_orbs[:,idx] = np.ndarray.flatten(np.dot(ref_mo[:,lumo_idx:], orbital))
   
    #print("  Energy scalings negative: ", scaling)
    #print("  Exchange matrix negative definite: ", neg_defin) 
    return ab2_orbs

def update_vir_orbs(wfn, lumo_idx, ab2_orbs):
    # Reconstruct virtual MO basis 
    if ab2_orbs.size == wfn.nbsf: 
         no_ab2 = 1
         ab2_orbs = ab2_orbs.reshape(-1,1) 
    else:     
         no_ab2 =  ab2_orbs.shape[1]    
    
    vir = np.zeros((wfn.nbsf, no_ab2 + wfn.nmo - lumo_idx), dtype = float)
    vir[:,:no_ab2] = ab2_orbs  
    vir[:, no_ab2:] = wfn.mo_coeff[:,lumo_idx:].copy() 
    
    # Remove linear dependencies and re-orthogonalise
    ortho_vir = orthogonalise(vir, wfn.integrals.overlap_matrix(), fill=False, lindep=True) 
    sys.stdout.flush()
     
    # Update wfn 
    wfn.mo_coeff[:,lumo_idx:] = ortho_vir 
    wfn.update()



#######################################################################
   
def get_comb_ab2_orbs(wfn):
    """
    Implementation of method to construct 1:1 Antibonding orbitals as proposed in (Aldossary, 2022)
    """
    print("\n")
    print("\n=============================")
    print(" AB2 Construction")
    print("=============================")
    
    # Localise and select occupied orbitals 
    local_coeff, ds = local_orbs(wfn)  
    bond_indices = np.argwhere(ds>1.5).flatten()
    
    # Update wfn and Fock matrices 
    wfn.mo_coeff[:,:wfn.nocc] = local_coeff
    wfn.update()  
    # Store Fock matrix in new MO basis 
    fock_mo = np.linalg.multi_dot((wfn.mo_coeff.T, wfn.fock, wfn.mo_coeff))
    
    # Compute AB2 orbitals 
    ab2_orbs = np.zeros((wfn.nbsf,2))
    
    # Construct the MO exchange matrix with localised occupied orbitals  
    exchange = np.zeros((wfn.nmo-wfn.nocc, wfn.nmo-wfn.nocc), dtype=float)
    for idx in [3,4]:
        dens = np.outer(local_coeff[:,idx],local_coeff[:,idx].T)
    
        ao_exchange = np.einsum('pqrs,qs->pr', wfn.integrals.tei_array(True, False) , dens, optimize="optimal")
        vir_mo_exchange = np.linalg.multi_dot((wfn.mo_coeff[:,wfn.nocc:].T, ao_exchange, wfn.mo_coeff[:,wfn.nocc:]))
    
        # Scale with MP2 denominator
        scaling = True
        neg_defin = True 
        for va in range(wfn.nocc, wfn.nmo): 
            for vb in range(wfn.nocc, wfn.nmo):
                delta = - fock_mo[va,va] - fock_mo[vb,vb] + 2*fock_mo[idx,idx]
                vir_mo_exchange[va - wfn.nocc, vb - wfn.nocc] /= delta
                if delta > 0: 
                    scaling = False
        exchange += vir_mo_exchange 
    
    # Compute and store lowest eigenvector  
    eigs, eigvecs = scipy.linalg.eigh(exchange)
    if np.any(eigs > 0): 
        neg_defin = False 

    # Select MO and Transform back into AO basis
    orb1 = eigvecs[:,0]
    orb2 = eigvecs[:,1]
     
    ab2_orbs[:,0] = np.ndarray.flatten(np.dot(wfn.mo_coeff[:,wfn.nocc:], orb1))
    ab2_orbs[:,1] = np.ndarray.flatten(np.dot(wfn.mo_coeff[:,wfn.nocc:], orb2))
    
    print("Energy scalings negative: ", scaling)
    print("Exchange matrix negative definite: ", neg_defin) 
    sys.stdout.flush()
    
    # Localise these orbitals
    pm = pipek.PM(wfn.integrals.mol,ab2_orbs)
    pm.pop_method = "mulliken" 
    _ = pm.kernel()
    
    # Perform Stability analysis 
    isstable, local_ab2  = pm.stability_jacobi()
    print("Stability analysis: ", isstable)
    if not isstable:
        _ = pm.kernel(local_ab2)
        isstable, local_ab2  = pm.stability_jacobi()
        print("Stability analysis two: ", isstable)
        assert( isstable ) 
    sys.stdout.flush() 
    
    return local_coeff, local_ab2, bond_indices
