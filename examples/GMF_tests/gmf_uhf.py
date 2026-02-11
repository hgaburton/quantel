import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.uhf import UHF

# Test UHF object with a range of optimisers
print("Test UHF object with a range of optimisers")

# Setup molecule and integrals
mol  = PySCFMolecule("formaldehyde.xyz", "6-31g", "angstrom")
ints = PySCFIntegrals(mol) # so here we dont put an exchange correlation functional in so it doesnt matter...

# Initialise UHF object
wfn = UHF(ints)

# Find ground state
from quantel.opt.lbfgs import LBFGS
wfn.get_orbital_guess(method="gwh")
LBFGS().run(wfn, plev=1)
wfn.get_davidson_hessian_index()
# Energy 
ground_energy = wfn.energy

# Generate initial guess from ground state 
wfn.excite()
from quantel.opt.test_gmf import GMF
GMF().run(wfn,plev=1, index=1)
# Canonicalize and compute Hessian index
wfn.canonicalize()
wfn.get_davidson_hessian_index()
# Energy gap 
print("=============")
print("Energy gap: ", wfn.energy-ground_energy)
print("=============")
