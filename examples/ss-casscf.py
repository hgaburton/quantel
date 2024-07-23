import os
# NOTE: there is some issue with inherent numpy multithreading (not fully diagnosed yet)
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np
import quantel
# Wave function method
from quantel.wfn.ss_casscf import SS_CASSCF
# Optimisation strategy (this is full second-order)
from quantel.opt.eigenvector_following import EigenFollow

np.set_printoptions(linewidth=10000,precision=6,suppress=True)
np.random.seed(7)

### Defining molecule and Libint2 interface ###
# Initialise molecular structure. This always requires the definition of units
mol = quantel.Molecule([["H",0.0,0.0,0.0],
                        ["H",0.0,0.0,1.0],
                        ["H",0.0,0.0,2.0],
                        ["H",0.0,0.0,3.0],
                        ["H",0.0,0.0,4.0],
                        ["H",0.0,0.0,5.0],
                        ["H",0.0,0.0,6.0],
                        ["H",0.0,1.0,7.0]], 
                       'bohr')  # bohr/angstrom (no default)
mol.print()

# Initialise interface to Libint2
ints = quantel.LibintInterface("6-31g",mol)

# Get the overlap matrix or oei 
ovlp = ints.overlap_matrix()
oei  = ints.oei_matrix()


### Define and optimise SS_CASSCF object ###
# Initialise RHF object from integrals
wfn = SS_CASSCF(ints,(6,(3,3)))

# The initialise method will automatically orthogonalise orbitals
mo_guess = np.random.rand(wfn.nmo,wfn.nmo)
ci_guess = np.random.rand(wfn.ndet,wfn.ndet)
wfn.initialise(mo_guess,ci_guess)

# Run eigenvector-following to target a minimum
EigenFollow().run(wfn, index=1)

# Print some information about optimal solution
print(f"\nEnergy = {wfn.energy: 16.10f}")
print(f" <S^2> = {wfn.s2: 9.3f}")

# Canonicalize the active space
wfn.canonicalize()

# Access the orbital coefficients in AO basis
mo_coeff = wfn.mo_coeff
# Access CI vector in active space (current state is always first column)
mat_ci   = wfn.mat_ci[:,0]

### I/O options ###
# Save the output to disk with tag '0001'
wfn.save_to_disk('0001')
wfn.deallocate()
del wfn

# Read wave function from disk
wfn2 = SS_CASSCF(ints,(6,(3,3)))
wfn2.read_from_disk('0001')
# Print some information about optimal solution
print(f"\nEnergy = {wfn2.energy: 16.10f}")
print(f" <S^2> = {wfn2.s2: 9.3f}")
