import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from pyscf.tools import cubegen
from quantel.wfn.rhf import RHF
from quantel.opt.lbfgs import LBFGS
from quantel.utils.ab2_orbitals import get_comb_ab2_orbs, update_vir_orbs 
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
local_coeff, ab2_orbs, bond_indices = get_comb_ab2_orbs(wfn)

cubegen.orbital(wfn.integrals.mol, "ab2.mo.3.cube", ab2_orbs[:,0])
cubegen.orbital(wfn.integrals.mol, "ab2.mo.4.cube", ab2_orbs[:,1])

cubegen.orbital(wfn.integrals.mol, "local.mo.3.cube", local_coeff[:,3])
cubegen.orbital(wfn.integrals.mol, "local.mo.4.cube", local_coeff[:,4])

update_vir_orbs(wfn, ab2_orbs[:,:])
wfn.mo_cubegen([wfn.nocc, wfn.nocc+1, wfn.nocc +2, wfn.nocc +3],fname="ovir") 

