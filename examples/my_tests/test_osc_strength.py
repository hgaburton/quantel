import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from pyscf.tools import cubegen 
from quantel.wfn.csf import CSF
from quantel.wfn.rhf import RHF
from quantel.opt.lbfgs import LBFGS
from quantel.opt.hybrid_ef import HybridEF
from quantel.drivers.noci import oscillator_strength

wfnlist=[]
# Initialise Molecule
mol  = PySCFMolecule("mol/formaldehyde.xyz" , "6-31g", "angstrom")
ints = PySCFIntegrals(mol)

# Input these into a CSF calculation 
wfn = CSF(ints, "+-")
wfn.get_orbital_guess(method="gwh") 

for ind in range(4): 
    HybridEF().run(wfn,index=ind, approx_hess=False, plev=1) 
    wfn.get_davidson_hessian_index(approx_hess=False)
    wfnlist.append(wfn.copy())

oscillator_strength(wfnlist) 
