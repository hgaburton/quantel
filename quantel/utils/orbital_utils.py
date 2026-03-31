from pyscf import scf, lo
import numpy as np 

def localise_orbitals(pymol,coeff,pop_method='becke'):
    """ 
       Wrapper for PYSCF orbital localisation via Pipek-Mezey scheme
       Args:
            pymol      : PySCF molecule object
            coeff      : Molecular orbital coefficients
            pop_method : Method for defining local regions
    """
    pm = lo.PM(pymol, coeff, scf.ROHF(pymol))
    pm.pop_method = pop_method
    coeff = pm.kernel()
    # Check stability of the localisation
    isstable, mo1 = pm.stability_jacobi()
    if not isstable:
        _ = pm.kernel(mo1)
        isstable, coeff  = pm.stability_jacobi()
    return coeff, isstable


def compute_mulliken_atomic_pops(pymol,coeff, give_bond_indices=False):    
    """
       Function to compute mulliken atomic populations to select orbitals 
       with bonding character. 
       Args:
            pymol      : PySCF molecule object
            coeff      : Molecular orbital coefficients
    """
    ds =[]
    if len(coeff.shape)==1: 
        pop, chg = scf.hf.mulliken_pop(mol=pymol, dm=np.outer(coeff[:,idx], coeff[:,idx]), verbose=0)
        chg += - py.mol.atom_charges()
        ds.append(1/np.sum(chg**2))
    else:
        for idx in range(coeff.shape[1]): 
            pop, chg = scf.hf.mulliken_pop(mol=pymol, dm=np.outer(coeff[:,idx], coeff[:,idx]), verbose=0)
            chg += - pymol.atom_charges()
            ds.append(1/np.sum(chg**2))
    
    ds = np.array(ds, dtype=float)  
    if give_bond_indices:
        return np.array(ds), np.argwhere(ds>1.5).flatten() 
    else:
        return np.array(ds) 
