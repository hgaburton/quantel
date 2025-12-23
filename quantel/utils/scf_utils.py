import numpy as np

def mom_select(Cocc, Cnew, metric):
    """ Select new occupied orbital coefficients using MOM criteria 
        Args:
            Cold : Previous set of occupied orbital coefficients 
            Cnew : New set of orbital coefficients from Fock diagonalisation
        Returns:
            Cnew reordered according to MOM criterion
    # Compute projections onto previous occupied space 
    """
    p = np.einsum('pj,pq,ql->l', Cocc,metric,Cnew,optimize="optimal")
    # Order MOs according to largest projection 
    idx = list(reversed(np.argsort(np.abs(p))))
    return Cnew[:,idx]