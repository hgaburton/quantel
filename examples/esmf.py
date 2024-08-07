import quantel
import numpy as np
from quantel.wfn.esmf import ESMF
from quantel.opt.eigenvector_following import EigenFollow

np.set_printoptions(linewidth=10000,precision=6,suppress=True)
np.random.seed(7)

# Initialise molecular structure (square H4)
mol = quantel.Molecule([["H",0.0,0.0,0.0],
                        ["H",0.0,1.0,0.0],
                        ["H",0.0,0.0,1.0],
                        ["H",0.0,1.0,1.0]])
print(mol.natom())
mol.print()

# Initialise interface to Libint2
ints = quantel.LibintInterface("6-31g",mol)
print("Overlap matrix in AO basis:")
print(ints.overlap_matrix())

# Initialise RHF object from integrals
wfn = ESMF(ints)

# The initialise method will automatically orthogonalise orbitals
mo_guess = np.random.rand(wfn.nbsf, wfn.nmo)
ci_guess = np.random.rand(wfn.ndet, wfn.ndet)
wfn.initialise(mo_guess,ci_guess)

# Run eigenvector-following to target a minimum
EigenFollow().run(wfn, index=0)

# Psuedo-canonicalise the result
wfn.canonicalize()

# Print some information about optimal solution
print(f"\nEnergy = {wfn.energy: 16.10f}")

print("\nMO coefficients:")
print(wfn.mo_coeff)

quit()
# Save the output to disk with tag '0001'
wfn.save_to_disk('0001')

C = wfn.mo_coeff
hcore = ints.oei_matrix(True)
print(ints.oei_ao_to_mo(C,C,True))
print(np.linalg.multi_dot([C.T, hcore, C]))
