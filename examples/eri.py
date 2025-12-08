import quantel
import numpy as np
from quantel.wfn.rhf import RHF
from quantel.wfn.cisolver import FCI
from quantel.opt.lbfgs import LBFGS
import datetime

np.set_printoptions(linewidth=10000,precision=6,suppress=True)
np.random.seed(7)

# Initialise molecular structure (square H4)
R = 9
mol = quantel.Molecule([["H",0.0,0.0,0.0*R],
                        ["H",0.0,0.0,1.0*R],
                        ["H",0.0,0.0,2.0*R],
                        ["H",0.0,0.0,3.0*R],
                        ["H",0.0,0.0,4.0*R],
                        ["H",0.0,0.0,5.0*R]
                       ],"bohr")
print("Molecule:")
mol.print()

# Initialise interface to Libint2
ints = quantel.LibintInterface("6-31g", mol)

# Initialise RHF object from integrals
wfn = RHF(ints)
wfn.get_orbital_guess()

# Find the RHF minimum
LBFGS().run(wfn)

# Setup the MO integral space
mo_ints = quantel.MOintegrals(ints)
# Argument order is coeff, ncore, nact
mo_ints.update_orbitals(wfn.mo_coeff.copy(),0,ints.nmo())

# This is in <pq|rs>
eri = mo_ints.tei_array(True,False)
print(np.linalg.norm(eri))

n = mo_ints.nmo()
n2 = n * n
V1 = np.zeros((n2,n2))
V2 = np.zeros((n2,n2))
for p in range(n):
    for q in range(n):
        for r in range(n):
            for s in range(n):
                V1[p*n+q,r*n+s] = eri[p,q,r,s]
                V2[p*n+q,r*n+s] = eri[p,r,q,s]

print("V1 = Vpq,rs = <pq|rs> -> <pq|pq>")
print(V1)
print("V2 = Vpq,rs = <pr|qs> -> <pp|qq>")
print(V2)
print(np.linalg.norm(V1))
print(np.linalg.norm(V2))

u, v = np.linalg.eigh(V1)
print(np.sort(u))
u, v = np.linalg.eigh(V2)
print(u)


print("Dominate paired doubles <pp|qq>")
Apq = np.einsum('ppqq->pq',eri)
print(Apq)
print(np.linalg.norm(Apq))
print("Dominate Jastrow <pq|pq>")
Bpq = np.einsum('pqpq->pq',eri)
print(Bpq)
print(np.linalg.norm(Bpq))
