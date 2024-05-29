import os
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import quantel
import numpy as np
from quantel.wfn.ss_casscf import SS_CASSCF
from quantel.opt.eigenvector_following import EigenFollow



np.set_printoptions(linewidth=10000,precision=6,suppress=True)
np.random.seed(7)

# Initialise molecular structure (square H4)
mol = quantel.Molecule([["H",0.0,0.0,0.0],
                        ["H",0.0,0.0,1.0],
                        ["H",0.0,0.0,2.0],
                        ["H",0.0,0.0,3.0],
                        ["H",0.0,0.0,4.0],
                        ["H",0.0,0.0,5.0],
                        ["H",0.0,0.0,6.0],
                        ["H",0.0,1.0,7.0]])
mol.print()

# Initialise interface to Libint2
ints = quantel.LibintInterface("6-31g",mol)

# Initialise RHF object from integrals
wfn = SS_CASSCF(ints,(6,(3,3)))

# The initialise method will automatically orthogonalise orbitals
np.random.seed(7)
mo_guess = np.random.rand(wfn.nmo,wfn.nmo)
ci_guess = np.random.rand(wfn.ndet,wfn.ndet)
wfn.initialise(mo_guess,ci_guess)

# Run eigenvector-following to target a minimum
EigenFollow().run(wfn, index=1)

# Print some information about optimal solution
print(f"\nEnergy = {wfn.energy: 16.10f}")
print(f" <S^2> = {wfn.s2: 9.3f}")


wfn.canonicalize()
# Save the output to disk with tag '0001'
wfn.save_to_disk('0001')
wfn.deallocate()
del wfn

wfn2 = SS_CASSCF(ints,(6,(3,3)))
wfn2.read_from_disk('0001')
# Print some information about optimal solution
print(f"\nEnergy = {wfn2.energy: 16.10f}")
print(f" <S^2> = {wfn2.s2: 9.3f}")
quit()

C = wfn.mo_coeff
hcore = ints.oei_matrix(True)
print(ints.oei_ao_to_mo(C,C,True))
print(np.linalg.multi_dot([C.T, hcore, C]))
