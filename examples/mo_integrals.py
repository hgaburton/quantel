import time
import quantel
import numpy as np
from quantel.wfn.rhf import RHF
from quantel.opt.eigenvector_following import EigenFollow

np.set_printoptions(linewidth=10000,precision=6,suppress=True)
np.random.seed(7)

# Initialise molecular structure (square H4)
mol = quantel.Molecule([["H",0.0,0.0,0.0],
                        ["H",0.0,0.0,1.0],
                        ["H",0.0,0.0,2.0],
                        ["H",0.0,0.0,3.0]])
print(mol.natom())
mol.print()

# Initialise interface to Libint2
ints = quantel.LibintInterface("cc-pvdz",mol)

# Initialise RHF object from integrals
wfn = RHF(ints)
wfn.get_orbital_guess()

# Run eigenvector-following to target a minimum
EigenFollow().run(wfn, index=0)
# Print some information about optimal solution
print(f"\nEnergy = {wfn.energy: 16.10f}")

print("\nMO coefficients:")
print(wfn.mo_coeff)

print("\nOrbital energies:")
print(wfn.orbital_energies)

print("\nDensity matrix in AO basis:")
print(wfn.dens)

print("\nFock matrix in AO basis:")
print(wfn.fock)

# Perform MO transfrom
C = wfn.mo_coeff.copy()
print("Initialise mo_ints object")
mo_ints = quantel.MOintegrals(C,C,ints)

h_ao = ints.oei_matrix(True)
start = time.time()
h_mo = np.linalg.multi_dot([C.T, h_ao, C])
mid = time.time()
print(h_mo)
print(mid - start)

mid = time.time()
h_mo = mo_ints.oei_matrix(True)
end = time.time()
print(h_mo)
print(end - mid)

