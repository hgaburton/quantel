import os
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import quantel
import numpy as np
from quantel.wfn.rhf import RHF
from quantel.opt.eigenvector_following import EigenFollow
from pygnme import utils
import time


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
mol = quantel.Molecule("mol.xyz","bohr")
print(mol.natom())
mol.print()

# Initialise interface to Libint2
ints = quantel.LibintInterface("sto-3g",mol)
print("Overlap matrix in AO basis:")
print(ints.overlap_matrix())

# Initialise RHF object from integrals
wfn = RHF(ints)

# The initialise method will automatically orthogonalise orbitals
mo_guess = np.random.rand(wfn.nbsf, wfn.nmo)
wfn.initialise(mo_guess)

# Run eigenvector-following to target a minimum
EigenFollow().run(wfn, index=0)

# Psuedo-canonicalise the result
wfn.canonicalize()

# Print some information about optimal solution
print(f"\nEnergy = {wfn.energy: 16.10f}")

#print("\nMO coefficients:")
#print(wfn.mo_coeff)
#
#print("\nOrbital energies:")
#print(wfn.orbital_energies)
#
#print("\nDensity matrix in AO basis:")
#print(wfn.dens)
#
#print("\nFock matrix in AO basis:")
#print(wfn.fock)

# Save the output to disk with tag '0001'
wfn.save_to_disk('0001')

mo_guess = wfn.mo_coeff.copy()

from quantel.wfn.csf import GenealogicalCSF

#print("RHF test")
#for i in range(5):
#    rhf = RHF(ints)
#    mo_guess = np.random.rand(wfn.nbsf, wfn.nmo)
#    rhf.initialise(mo_guess)
#    #EigenFollow().run(rhf, index=0)
#    rhf.canonicalize()
#    print(rhf.nocc)
#    print(rhf.mo_coeff[:,:rhf.nocc])
#    del rhf

print("CSF test")
for val in [True]:
    csf = GenealogicalCSF(ints,'+-',nohess=val)
    mo_guess = np.random.rand(wfn.nbsf, wfn.nmo)
    csf.initialise(mo_guess,'++--')
    start = time.time()
    csf.update_integrals()
    print(csf.gradient)
    print(csf.get_numerical_gradient())
    end = time.time()
    print(end-start)
quit()
print("Number of determinants:", csf.ndet)
#EigenFollow().run(csf, index=0)
print(csf.mo_coeff[:,csf.ncore:csf.nocc])
csf.save_to_disk('0001')
del csf

csf = GenealogicalCSF(ints)
csf.read_from_disk('0001')
EigenFollow().run(csf, index=0)
