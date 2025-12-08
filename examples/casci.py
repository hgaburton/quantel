import quantel
import numpy as np
from quantel.wfn.csf import CSF
from quantel.wfn.cisolver import FCI
from quantel.opt.lbfgs import LBFGS
import datetime

np.set_printoptions(linewidth=10000,precision=6,suppress=True)
np.random.seed(7)

# Initialise molecular structure (square H4)
R = 3
mol = quantel.Molecule([["H",0.0,0.0,0.0*R],
                        ["He",0.0,0.0,1.0*R]
                        ["H",0.0,0.0,1.0*R]
                       ],"bohr")
print("Molecule:")
mol.print()

# Initialise interface to Libint2
ints = quantel.LibintInterface("6-31g", mol)

# Initialise RHF object from integrals
wfn = CSF(ints,"+-")
wfn.get_orbital_guess()

# Find the RHF minimum
LBFGS().run(wfn)

wfn.write_fcidump('0001')
wfn.write_cidump('0001')

# Setup the MO integral space
mo_ints = quantel.MOintegrals(ints)
# Argument order is coeff, ncore, nact
mo_ints.update_orbitals(wfn.mo_coeff.copy(),0,ints.nmo())

# Setup FCI solver object
nelec = (mol.nalfa(), mol.nbeta())
fci = FCI(mo_ints, nelec)

# Solve for nroots = 5
nroots = 5
e,x = fci.solve(nroots,verbose=5)

# Print the converged eigenvalues
print()
print(" FCI dimension = ", fci.ndet)
print(" Converged eigenvalues (Eh):")
for ev in e:
    print(f"  {ev: 16.10f}")
