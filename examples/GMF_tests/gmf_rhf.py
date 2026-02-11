import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.opt.lbfgs import LBFGS
#from quantel.opt.gmf import GMF
from quantel.opt.test_gmf import GMF
from quantel.wfn.rhf import RHF

molxyz = "formaldehyde.xyz"

# Test RSF object with GMF optimisation 
#mol  = PySCFMolecule(molxyz, "aug-ccpvtz", "angstrom")
mol  = PySCFMolecule(molxyz, "6-31g", "angstrom")
ints = PySCFIntegrals(mol)

# Initialise RHF object 
wfn = RHF(ints)  

# Setup optimiser
wfn.get_orbital_guess(method="gwh")
LBFGS().run(wfn)
wfn.canonicalize()
wfn.get_davidson_hessian_index()
ground_energy = wfn.energy

# Generate initial guess from ground state 
wfn.excite()
GMF().run(wfn,plev=1, index=3)
# Canonicalize and compute Hessian index
wfn.canonicalize()
wfn.get_davidson_hessian_index()
# Energy gap 
print("=============")
print("Energy gap: ", wfn.energy-ground_energy)
print("=============")


