import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from pyscf.tools import cubegen 
from quantel.wfn.csf import CSF
from quantel.wfn.rhf import RHF
from quantel.opt.lbfgs import LBFGS
from quantel.opt.test_gmf import GMF

# CSF with RHF excited orbtials
# Initialise Molecule
molxyz = "formaldehyde.xyz"
mol  = PySCFMolecule(molxyz, "6-31g", "angstrom")
ints = PySCFIntegrals(mol)


# Optimise onto RHF ground state 
ground_wfn = RHF(ints)
ground_wfn.get_orbital_guess(method="gwh")
LBFGS().run(ground_wfn)
ground_wfn.canonicalize()
ground_wfn.get_davidson_hessian_index()

ground_wfn.mo_cubegen([ground_wfn.nocc -1,ground_wfn.nocc],fname="start") 

# Input these into a CSF calculation 
csf_wfn = CSF(ints, "+-")
csf_wfn.initialise(mo_guess=ground_wfn.mo_coeff)

# Perfom Saddle point optimisation  
GMF().run(csf_wfn,plev=1, index=1)

# Canonicalise and obtain Hessian index 
csf_wfn.canonicalize()
csf_wfn.get_davidson_hessian_index()

# This gives us the singly occupied orbitals
csf_wfn.mo_cubegen([7,8],"end")

DeltaE = csf_wfn.energy - ground_wfn.energy 
print("\n========================")
print("DeltaE: ", DeltaE)



