import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.csf import CSF
from quantel.wfn.rhf import RHF
from quantel.opt.lbfgs import LBFGS
from quantel.opt.test_gmf import GMF

# CSF with RHF excited orbtials
# Initialise Molecule
molxyz = "formaldehyde.xyz"
mol  = PySCFMolecule(molxyz, "aug-cc-pvdz", "angstrom")
ints = PySCFIntegrals(mol)

# Perform ground state optimisation 
ground_wfn = RHF(ints)
ground_wfn.get_orbital_guess(method="gwh")
LBFGS().run(ground_wfn)

# Canonicalise and obtain Hessian index 
ground_wfn.canonicalize()
ground_wfn.get_davidson_hessian_index()
ground_wfn.mo_cubegen([5,6,7,8,9,10,11,12],"ground_rhf")


