import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from pyscf.tools import cubegen 
from quantel.wfn.csf import CSF
from quantel.wfn.rhf import RHF
from quantel.opt.lbfgs import LBFGS
from quantel.opt.test_gmf import GMF

from quantel.utils.ab2_orbitals import get_ab2_orbs, update_vir_orbs  
from quantel.utils.linalg import matrix_print  

# CSF with RHF excited orbtials
# Initialise Molecule
molxyz = "formaldehyde.xyz"
#mol  = PySCFMolecule(molxyz, "aug-cc-pvdz", "angstrom")
mol  = PySCFMolecule(molxyz, "6-31g", "angstrom")
ints = PySCFIntegrals(mol)


# Optimise onto RHF ground state 
ground_wfn = RHF(ints)
ground_wfn.get_orbital_guess(method="gwh")
LBFGS().run(ground_wfn)
ground_wfn.canonicalize()
ground_wfn.get_davidson_hessian_index()

# Compute and save AB2 orbitals
local_coeff, ab2_orbs, bond_indices = get_ab2_orbs(ground_wfn)
#for idx, BO in enumerate(bond_indices): 
#    cubegen.orbital(ground_wfn.integrals.mol, f"ab2.mo.{BO}.cube", ab2_orbs[:,idx])
#    cubegen.orbital(ground_wfn.integrals.mol, f"local.mo.{BO}.cube", local_coeff[:,BO])

update_vir_orbs(ground_wfn, ab2_orbs[:,-1])
ground_wfn.mo_cubegen([ground_wfn.nocc -1,ground_wfn.nocc],fname="start") 

# Input these into a CSF calculation 
csf_wfn = CSF(ints, "+-")
csf_wfn.initialise(mo_guess=ground_wfn.mo_coeff)
print("number of occupied orbs: ",csf_wfn.nocc)
# Perfom Saddle point optimisation  
GMF().run(csf_wfn,plev=1, index=2, maxit=200)

# Canonicalise and obtain Hessian index 
print("\n===============================")
print("Analysing optimsed state") 
print("==============================") 
csf_wfn.canonicalize()

# is this too hard to do alone - best to use the previous guess?
csf_wfn.get_davidson_hessian_index()

# This gives us the singly occupied orbitals
csf_wfn.mo_cubegen([7,8],"end")

DeltaE = csf_wfn.energy - ground_wfn.energy 
print("\n========================")
print("DeltaE: ", DeltaE)



