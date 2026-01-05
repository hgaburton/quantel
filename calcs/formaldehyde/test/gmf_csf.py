import quantel
import numpy as np
import sys 

from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from pyscf.tools import cubegen 
from quantel.wfn.csf import CSF
from quantel.wfn.rhf import RHF
from quantel.opt.lbfgs import LBFGS
from quantel.opt.test_gmf import GMF
from quantel.utils.ab2_orbitals import get_ab2_orbs, update_vir_orbs  
from quantel.utils.linalg import matrix_print  

#check the number of arguments 
if(len(sys.argv[1:])==0): 
    # Initialise CSF with RHF excited orbtials
    mol  = PySCFMolecule("geom.xyz", "6-31g", "angstrom")
    ints = PySCFIntegrals(mol)

    # Optimise onto RHF ground state 
    ground_wfn = RHF(ints)
    ground_wfn.get_orbital_guess(method="gwh")
    LBFGS().run(ground_wfn)
    ground_wfn.canonicalize()
    ground_wfn.get_davidson_hessian_index()

    # Compute and save AB2 orbitals
    local_coeff, ab2_orbs, bond_indices = get_ab2_orbs(ground_wfn)

    update_vir_orbs(ground_wfn, ab2_orbs[:,-1])
    ground_wfn.save_to_disk('rhf_start') 

    # Input these into a CSF calculation 
    csf_wfn = CSF(ints, "+-")
    csf_wfn.initialise(mo_guess=ground_wfn.mo_coeff)

elif(len(sys.argv[1:])==1): 
    # Input are the read from disk orbitals
    csf_wfn = CSF.read_from_disk(sys.argv[1]) 
    # this input will need to be the path to the file without the .hdf5 target

# Perfom Saddle point optimisation  
GMF().run(csf_wfn,plev=1, index=2, maxit=200)

# Canonicalise and obtain Hessian index 
csf_wfn.canonicalize()
csf_wfn.get_davidson_hessian_index()

csf_wfn.save_to_disk("optimised_wfn")



