import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds 
from quantel.utils.linalg import matrix_print

def sorting_shells(wfn, Cnew, projections):
    """ This finds the optimal order for CSF wave functions - can we even do that? 
    - well we should be able to cause the R matrix is just a matrix of numbers and we get coefficients out as our eigenfunctions
    we arent going to update the fock matrix elements just yet so that means any order we put this in the R matrix will still be diagonal..

        - what is the form of the projections matrix, <new orb, abs(projection onto old shell) > - so its (num. orbs, wfn.nshell + 2 )

    """
    all_indices = [wfn.core_indices] + wfn.shell_indices + [[i for i in range(wfn.ncore + wfn.nopen, wfn.nmo)]] 
    #
    shell_pops = [len(i) for i in all_indices]
    # Def matrix of projections of each new orbital onto each occupied shell - with virtual shell projections defined to be zero
    x = np.zeros((wfn.nmo, wfn.nshell+2))
    x[:,:-1] = projections 
    
    #######
    num_items, num_groups = wfn.nmo, (wfn.nshell+2)
    num_vars = num_items*num_groups
    # scipy find the minimum - just flip it over to find the max
    c = -x.flatten() #flatten goes row by row
    # Each item assigned to a single group 
    A_item = np.zeros((num_items, num_vars)) 
    for i in range(num_items): 
        for g in range(num_groups): 
            A_item[i,i*num_groups+g]= 1 ##so for each row of the A matrix all of the y[i,0]=y[i,1]=y[i,2]=1 
    
    # Group size constraints
    A_group = np.zeros((num_groups, num_vars)) 
    for g in range(num_groups): 
        for i in range(num_items): 
            A_group[g, i*num_groups+g] = 1
    # 
    A = np.vstack([A_item, A_group])
    b = np.concatenate([np.ones(num_items), np.array(shell_pops)]) 
    constraints = LinearConstraint(A,b,b)
    bounds=Bounds(0,1)
    integrality = np.ones(num_vars, dtype=int)
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    assignment = res.x.reshape(num_items, num_groups) 
    group_of_item = assignment.argmax(axis=1) 
    # group of item is a list with group index assignment
    order=[]
    for i in range(wfn.nshell+2): 
        indices = [ind for ind, val in enumerate(group_of_item) if val == i ] 
        order.append(indices) 
    ####
    order = [i for shell in order for i in shell ]
    Cnew = Cnew[:, order ]
    return Cnew, order  


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







