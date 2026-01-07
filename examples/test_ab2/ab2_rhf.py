import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from pyscf.tools import cubegen
from quantel.wfn.rhf import RHF
from quantel.opt.lbfgs import LBFGS
from quantel.utils.ab2_orbitals import get_ab2_orbs, update_vir_orbs 
import sys 

# Test AB2 on RHF orbitals  
print("Test AB2 on RHF orbitals")

mol  = PySCFMolecule("formaldehyde.xyz", "6-31g", "angstrom")
ints = PySCFIntegrals(mol)

# Optimise onto RHF ground state 
wfn = RHF(ints)
wfn.get_orbital_guess(method="gwh")
LBFGS().run(wfn)
wfn.canonicalize()
wfn.get_davidson_hessian_index()

# Compute and save AB2 orbitals
ab2_orbs, bond_indices = get_ab2_orbs(wfn)
for idx, BO in enumerate(bond_indices): 
    cubegen.orbital(wfn.integrals.mol, f"ab2.mo.{BO}.cube", ab2_orbs[:,idx])

update_vir_orbs(wfn, ab2_orbs[:,:])
wfn.mo_cubegen([8,9,10,11,12,13,14,15,16,17,18,19,20,21],fname="ovir") 

