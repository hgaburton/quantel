import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.uhf import UHF
from quantel.opt.diis import DIIS
from quantel.opt.lbfgs import LBFGS
import copy 

print("\n=============================================================")
print("Testing (I)MOM-DIIS on UHF object for excited state optimisation")
print("=============================================================")

# Setup molecule and integrals
mol  = PySCFMolecule("formaldehyde.xyz", "6-31g", "angstrom")
ints = PySCFIntegrals(mol) 

# Initialise UHF object and optimise onto ground state 
wfn = UHF(ints)
wfn.get_orbital_guess(method="core", asymmetric=True)
LBFGS().run(wfn,plev= 1)
wfn.canonicalize()
wfn.get_davidson_hessian_index()

# Generate MO cubefiles 
#wfn.mo_cubegen([wfn.alfa.nocc-1, wfn.alfa.nocc], [wfn.beta.nocc-1,wfn.beta.nocc],fname="ground")

for selector in ["mom", "imom"]:
    # Just do an alfa excitation
    excite_wfn = wfn.excite(occ_idx=[[wfn.nocc[0]-1],[]],vir_idx=[[wfn.nocc[0]],[]],mom_method=selector)
    # Perform MOM optimisation
    DIIS().run(excite_wfn,plev= 1)
    excite_wfn.canonicalize()
    excite_wfn.get_davidson_hessian_index(approx_hess=False)
    excite_wfn.print()
    
    # Generate MO cubefiles 
    #excite_wfn.mo_cubegen([wfn.alfa.nocc-1, wfn.alfa.nocc], [wfn.beta.nocc-1,wfn.beta.nocc],fname="exite"+selector)
    
    # Compute energy gap 
    print("===================")
    print("Energy gap: ", excite_wfn.energy - wfn.energy)
    print("===================")





