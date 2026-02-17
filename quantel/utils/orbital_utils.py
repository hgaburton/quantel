from pyscf import scf, lo

def localise_orbitals(pymol,coeff,pop_method='becke'):
    """ Localise orbitals using Pipek-Mezey localisation
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
        coeff = pm.kernel(mo1)
    return coeff
