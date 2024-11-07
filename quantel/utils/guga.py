import numpy as np

def A(b,x,y):
    return np.sqrt((b+x)/(b+y))

def F(b,d):
    if(d == 0 or d == 3):
        return 1
    elif(d == 1):
        return A(b,2,0) * A(b,-1,1)
    elif(d == 2):
        return A(b,0,2) * A(b,3,1)

def e_ijji(b,d,i,j):
    """ Compute exchange coupling term for CSF using b vector and drt vector
            :param b: b vector
            :param d: Distinct row table vector
            :param i: index i
            :param j: index j
            :return: value
    """
    if(np.any(d > 3)):
        raise RuntimeError('Invalid distinct row table vector')
    
    # Want to make sure i < j
    swap = i > j
    if(i > j):
        i,j = j,i
        
    if(i == j):
        return -1
    
    # Compute the value
    value = 0
    if(d[i] == 0 or d[j] == 0):
        value = 0
    
    elif(d[i] == 3):
        if(d[j] == 1 or d[j] == 2):
            value = -1
        elif(d[j] == 3):
            value = -2
        
    elif(d[j] == 3):
        if(d[i] == 1 or d[i] == 2):
            value = -1
        elif(d[i] == 3):
            value = -2
    
    else:
        prod  = (A(b[i],2,0) if d[i] == 1 else A(b[i],0,2))
        prod *= (A(b[j],-1,1) if d[j] == 1 else A(b[j],3,1))
        for k in range(i+1,j):
            prod *= F(b[k],d[k])
        phase = 1 if d[i] == d[j] else -1
        value = -0.5 * (1 + phase * prod)

    if(swap):
        i,j = j,i

    return value