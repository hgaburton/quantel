#!/usr/bin/env python

import quantel
import numpy as np
from quantel.wfn.rhf import RHF
from quantel.opt.diis import DIIS

np.set_printoptions(linewidth=1000,suppress=True,precision=6)

print(" *** NOTE dipole matrix does not include nuclear contribution *** ")

# Define our H2 geometry (angstrom)
R = 1.5
atoms = [["H", 0.0, 0.0,-0.5*R],
         ["H", 0.0, 0.0, 0.5*R]]

# Setup the Quantel object
mol = quantel.Molecule(atoms,"angstrom")

# Initialise interface to Libint2
ints = quantel.LibintInterface("6-31g",mol)

# Initialise RHF object from integrals
wfn = RHF(ints)
wfn.get_orbital_guess("gwh")

# Run DIIS to get RHF solutions
DIIS().run(wfn)

# Get cofeficients
coeff = wfn.mo_coeff.copy()

# Construct MO integral object
mo_ints = quantel.MOintegrals(ints)
mo_ints.update_orbitals(coeff,0,ints.nmo())

# Access dipole integrals in MO basis to inspect
print("\nDipole integrals in MO basis...")
dip_mo = mo_ints.dipole_matrix(True)
for (i,l) in enumerate(['x','y','z']):
    print(l)
    print(dip_mo[i])

# Verify dipole integrals using standard transform
dip_ao = ints.dipole_matrix()
print("\nDipole integrals in MO basis computed through numpy...")
for (i,l) in enumerate(['x','y','z']):
    print(l)
    print(coeff.T @ dip_ao[i] @ coeff)

# Build the FCI space
cispace = quantel.CIspace(mo_ints,ints.nmo(),1,1)
cispace.initialize('FCI')
# Print it so we see the determinant ordering
print("\nCI space includes...")
cispace.print()

# Compute and print dipole matrix and Hamiltonian in FCI basis
print("\nFCI Hamiltonian matrix...")
Hmat = cispace.build_Hmat()
print(Hmat)
print(f"Ground state energy (Eh) = {np.linalg.eigvalsh(Hmat)[0]: 10.8f}")

print("\nFCI dipole matrix (x-component)...")
Dmat = cispace.build_Dmat() 
print(Dmat[0])
