import os
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import quantel
import numpy as np
from quantel.wfn.rhf import RHF
from quantel.opt.eigenvector_following import EigenFollow

np.set_printoptions(linewidth=10000,precision=6,suppress=True)
np.random.seed(7)

s = 1.8897259886
r1 =  1.71640
r2 = -0.44026
r3 = -0.43884
r4 = -0.24377 
r5 = -0.97325
r6 =  0.95217

# Initialise molecular structure (square H4)
#mol = quantel.Molecule([["C", 0.0,      0.0,       0.0],
#                        ["O", 1.20900*s,0.0,       0.0],
#                        ["H",-0.59827*s,0.0,      -0.94019*s],
#                        ["H",-0.59827*s,0.00049*s, 0.94019*s]])
#mol = quantel.Molecule([["C", 0.0,       0.0,       0.0],
#                        ["O", 1.71640*s, 0.0,       0.0],
#                        ["H",-0.44026*s, 0.0,      -0.97325*s],
#                        ["H",-0.43884*s,-0.24377*s, 0.95217*s]])
mol = quantel.Molecule([["C", 0.0,  0.0,  0.0],
                        ["O", r1*s, 0.0,  0.0],
                        ["H", r2*s, 0.0,  r5*s],
                        ["H", r3*s, r4*s, r6*s]])
print(mol.natom())
mol.print()

# Initialise interface to Libint2
ints = quantel.LibintInterface("def2-svp",mol)
print("Overlap matrix in AO basis:")
print(ints.overlap_matrix())

# Initialise RHF object from integrals
wfn = RHF(ints)

# The initialise method will automatically orthogonalise orbitals
for trial in range(500):
    mo_guess = np.random.rand(wfn.nbsf, wfn.nmo)
    wfn.initialise(mo_guess)

# Run eigenvector-following to target a minimum
    EigenFollow().run(wfn, index=0)

# Psuedo-canonicalise the result
    wfn.canonicalize()

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

# Save the output to disk with tag '0001'
wfn.save_to_disk('0001')

C = wfn.mo_coeff
hcore = ints.oei_matrix(True)
print(ints.oei_ao_to_mo(C,C,True))
print(np.linalg.multi_dot([C.T, hcore, C]))
