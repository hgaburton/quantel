import quantel
import numpy as np
from quantel.wfn.csf import CSF
from quantel.opt.lbfgs import LBFGS

np.set_printoptions(linewidth=10000,precision=6,suppress=True)
np.random.seed(7)

# Initialise molecular structure (square H4)
R = 0.74
mol = quantel.Molecule([["H",0.0,0.0,-3*R],
                        ["H",0.0,0.0, 3*R],
                        ["He",0.0,0.0,0]
                       ],'angstrom')
no = mol.nalfa()
ncore = 0
nact = 6
ne = no - ncore
print(ne)
mol.print()

# Initialise interface to Libint2
ints = quantel.LibintInterface("3-21g",mol)

# Initialise RHF object from integrals
cguess = np.zeros((6,6))
cguess[4,0] = 1
cguess[0,1] = 1
cguess[2,2] = 1
cguess[5,3] = 1
cguess[1,4] = 1
cguess[3,5] = 1
wfn = CSF(ints,'+-')
wfn.initialise(cguess,'+-')
print(wfn.mo_coeff)

# Run eigenvector-following to target a minimum
LBFGS().run(wfn)

# Get number of MOs and coefficients
nmo = ints.nmo()
wfn.canonicalize()
C = wfn.mo_coeff.copy()
# Print some information about optimal solution
print(f"\nNuclear repulsion = {ints.scalar_potential(): 16.10f}")
print(f"Energy = {wfn.energy: 16.10f}")
print("\nMO coefficients:")
print(wfn.mo_coeff)
Fao = wfn.fock
Fmo = np.linalg.multi_dot([C.T, Fao, C])

# Construct MO integral objectPerform MO transfrom
print("Initialise MO integrals")
mo_ints = quantel.MOintegrals(ints)
print("Compute integrals")
mo_ints.update_orbitals(C,ncore,nact)
print("Done")

# Build CI object
print("Initialise CI object... ")
cispace = quantel.CIspace(mo_ints,nact,no-ncore,no-ncore)
detlist = ['2ab000','2ba000']
detlist = detlist + ['aabb00','abab00','baba00','bbaa00',
                     'aab0b0','aba0b0','bab0a0','bba0a0',
                     'aab00b','aba00b','bab00a','bba00a'] # ia
detlist = detlist + ['2a0b00','2b0a00','20ab00','20ba00',
                     '2a00b0','2b00a0','20a0b0','20b0a0',
                     '2a000b','2b000a','20a00b','20b00a'
                    ] # pa
detlist = detlist + ['220000','202000'] # pq
detlist = detlist + ['a2b000','b2a000','ab2000','ba2000'] # ip
detlist = detlist + ['abba00','baab00',
                     'abb0a0','baa0b0',
                     'abb00a','baa00b'
                    ] # ip pa
cispace.initialize('custom',detlist)
cispace.print()
print("done")

Hfci = cispace.build_Hmat()
e,v = np.linalg.eigh(Hfci)
print(e[:10])
for i in range(10):
    print(f"\nE(state {i}) = {e[i]-e[0]: 10.6f}")
    print("--------------------------------")
    cispace.print_vector(v[:,i],1e-2)
