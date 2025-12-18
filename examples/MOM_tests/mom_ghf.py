import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.ghf import GHF
from quantel.opt.diis import DIIS
from quantel.opt.lbfgs import LBFGS
import copy 

print("\n=============================================================")
print("Testing (I)MOM-DIIS on RHF object for excited state optimisation")
print("=============================================================")

# Setup molecule and integrals
mol  = PySCFMolecule("formaldehyde.xyz", "6-31g", "angstrom")
ints = PySCFIntegrals(mol) 

# Initialise RHF object and optimise onto ground state 
wfn = GHF(ints)
wfn.get_orbital_guess(method="core")
LBFGS().run(wfn,plev= 1)
wfn.canonicalize()
wfn.get_davidson_hessian_index()

## Generate MO cubefiles 
#wfn.mo_cubegen([wfn.alfa.nocc-1, wfn.alfa.nocc], [wfn.beta.nocc-1,wfn.beta.nocc],fname="ground")

for selector in ["mom", "imom"]:
    excite_wfn = copy.deepcopy(wfn)
    # Excite ground state  
    excite_wfn.excite()
    # Perform MOM optimisation
    DIIS(occupation_selector=selector).run(excite_wfn,plev= 1)
    excite_wfn.canonicalize()
    excite_wfn.get_davidson_hessian_index()
    
    ## Generate MO cubefiles 
    #wfn.mo_cubegen([wfn.nalfa-1, excite_wfn.nalfa],fname=selector)
    
    # Compute energy gap 
    print("===================")
    print("Energy gap: ", excite_wfn.energy - wfn.energy)
    print("===================")





