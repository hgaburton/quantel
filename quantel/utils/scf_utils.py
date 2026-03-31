import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds 
from quantel.utils.linalg import matrix_print

def shell_sort(wfn, Cnew, projections):
    """  
        Routine to compute the orbital ordering that maximise total shell projections, via 
        optimisation with respect to two linear contraints: number of occupied orbitals and 
        shell population.        
       
        Rough outline:  
        - x: vector of optimsation variables 
        - c: property 
        - A,b: constraints 
        
        Solves minimisation of c @ x w.r.t constraints A @ x = b
    """
    all_indices = [wfn.core_indices] + wfn.shell_indices + [[i for i in range(wfn.ncore + wfn.nopen, wfn.nmo)]] 
    shell_pops = [len(i) for i in all_indices]
    
    # Set up parameters for optimisation 
    num_items, num_groups = wfn.nmo, (wfn.nshell+2)
    num_vars = num_items*num_groups
   
    # All projections in vectorised format 
    c = np.zeros((wfn.nmo, wfn.nshell+2))
    c[:,:-1] = projections 
    c = -c.flatten() 
    
    # Constrain each orbital to a single shell
    # A_item @ x_item = b_item  
    A_item = np.zeros((num_items, num_vars)) 
    for i in range(num_items): 
        for g in range(num_groups): 
            A_item[i,i*num_groups+g]= 1
    b_item = np.ones(num_items) 

    # Constrain number of orbitals per shell
    # A_group @ x_group = b_group   
    A_group = np.zeros((num_groups, num_vars)) 
    for g in range(num_groups): 
        for i in range(num_items): 
            A_group[g, i*num_groups+g] = 1
    b_group = np.array(shell_pops) 

    # Combine contraints 
    A = np.vstack([A_item, A_group])
    b = np.concatenate([b_item, b_group]) 
    # Initialise LinearConstraint object 
    constraints = LinearConstraint(A,b,b)
    # Enforce values = 0 or 1 
    bounds=Bounds(0,1)
    integrality = np.ones(num_vars, dtype=int)
    # Solve via Mixed Integer linear program
    result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    assignment = result.x.reshape(num_items, num_groups) 
    group_of_item = assignment.argmax(axis=1) 
   
    # Extract optimal ordering  
    order=[]
    for i in range(wfn.nshell+2): 
        indices = [ind for ind, val in enumerate(group_of_item) if val == i ] 
        order.append(indices) 
     
    order = [i for shell in order for i in shell ]
    Cnew = Cnew[:, order ]
    return Cnew 


def mom_select(Cocc, Cnew, metric):
    """ Select new occupied orbital coefficients using MOM criteria - for closed shell case 
    """
    p = np.einsum('pj,pq,ql->l', Cocc,metric,Cnew,optimize="optimal")
    # Order MOs according to largest projection 
    idx = list(reversed(np.argsort(np.abs(p))))
    return Cnew[:,idx]
