from quantel.utils.csf_utils import get_vector_coupling, get_csf_vector, get_shells, get_shell_exchange, get_ensemble_expansion
from scipy.optimize import minimize
import numpy as np
from itertools import product, combinations

np.set_printoptions(linewidth=1000,precision=6,suppress=True)

def prod(A,B):
    return np.trace(A.T @ B)

def get_uhf_coupling(occ):
    n = len(occ)
    mat = np.zeros((n,n))
    for p,pchar in enumerate(occ):
        for q, qchar in enumerate(occ):
            if(pchar==qchar and p>q):
                mat[p,q] = -1
    mat[0,0] = 1
    return mat

def shell_subspace(m,shl):
    nshl=len(shl)
    mat = np.zeros((nshl,nshl))
    for I in range(nshl):
        for J in range(nshl):
            mat[I,J] = m[shl[I][0],shl[J][0]]
    return mat

print(get_vector_coupling(6,0,6,'+++---')[1])
print(get_vector_coupling(6,0,6,'+-+-+-')[1])

spin_coupling = '+++-+-+--'
print(spin_coupling)
a = get_ensemble_expansion(spin_coupling)
quit()

n=2

spin_coupling = '+-+-+'
for L in range(10):
  for spin_coupling in product('+-',repeat=L+1):
    spin_vec = [1 if s=='+' else -1 for s in spin_coupling]
    if(np.any(np.cumsum(spin_vec)<0)):
        continue
    spin_coupling = ''.join(spin_coupling)
   
    nopen = len(spin_coupling)
    #print(nopen)
    shells = get_shells(0,spin_coupling)[1]
    nshell = len(shells)
    #print(shells)
    beta = get_shell_exchange(0,shells,spin_coupling)
    #print("Shell exchange")
    #print(beta)
    b = get_vector_coupling(nopen,0,nopen,spin_coupling)[1]

    count=0
    for x,y in get_csf_vector(spin_coupling):
        if(abs(y)>1e-10):
            count+=1

    b = np.tril(b,k=-1)
    b[0,0] = 1
    #print(b)

# Construct basis with at most 2 shell spin flips
# We need to include the 'aaaaaa' determinant for normalisation
    if(nshell==4 or nshell<=2):
        strs=[]
        for it in combinations(range(0,len(shells)),r=1):
            occs = np.zeros(nopen)
            shell1 = shells[it[0]]
            occs[shell1]=1
            if(occs[0]==1):
                occs = 1-occs
            occstr = ''.join(['a' if(occi==0) else 'b' for occi in occs])
            if(not occstr in strs):
                strs.append(occstr)
    else:
        strs=['a'*nopen]

    for it in combinations(range(0,len(shells)),r=2):
        occs = np.zeros(nopen)
        shell1 = shells[it[0]]
        occs[shell1]=1
        shell2 = shells[it[1]]
        occs[shell2]=1
        if(occs[0]==1):
            occs = 1-occs
        occstr = ''.join(['a' if(occi==0) else 'b' for occi in occs])
        if(not occstr in strs):
            strs.append(occstr)
    strs.sort()
    #print(len(strs),int(nshell*(nshell-1)/2+1))

    #print("Basis with vectorised form")
    basis=[]
    for string in strs:#'abab']:#,'aaab','aaba','abaa']:
        m = get_uhf_coupling(string)
        basis.append(m.copy())
        #print(string)
        Mbeta = shell_subspace(m,shells)
        #print(Mbeta[np.tril_indices(nshell,k=-1)])

# Build basis metric tensor
    nb = len(basis)
    S = np.zeros((nb,nb))
    for i in range(nb):
        for j in range(nb):
            S[i,j] = prod(basis[i],basis[j])
    e,v = np.linalg.eigh(S)
    #print("Eigenvalues of basis metric tensor")
    #print(e)
    #print("Eigenvectors of basis metric tensor")
    #print(v)
# Build pseudo-inverse 
    for i in range(e.size):
        if abs(e[i]) > 1e-10:
            e[i] = 1/e[i]
    X = v @ np.diag(e) @ v.T

# Form the projection of target exchange matrix
    vec = X @ np.array([prod(basis[i],b) for i in range(nb)])
    #print(spin_coupling)
    #print("Exchange matrix expansion coefficients:")
    m = np.zeros(b.shape)
    for i in range(nb):
        #print(f"{strs[i]}: {vec[i]: 8.4f}")
        m += vec[i] * basis[i]

    #print("Target exchange matrix")
    #print(b)
    #print("Recovered exchange matrix")
    #print(m)
    print(f"{spin_coupling:15s}  Exchange error = {np.linalg.norm(m-b): 6.3e}  Emin={e[0]: 6.3e}  Nop={nb: 5d}")
    #print("Normalisation constraint = ",np.sum(vec))
