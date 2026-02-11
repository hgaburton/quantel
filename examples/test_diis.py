from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.opt.lbfgs import LBFGS
from quantel.opt.diis import DIIS
import numpy as np
from quantel.utils.linalg import matrix_print
np.set_printoptions(linewidth=1000,precision=6,suppress=True)

# Set Occupation selector 
mom_method = "MOM"

if __name__ == "__main__":
    print("\n===============================================")
    print(f" Testing (I/MOM)-DIIS opt. on CSF and ROKS objects")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule("mol/formaldehyde.xyz","aug-cc-pvdz", "angstrom")
    for method in ("csf", "roks"): 
        if method=="csf": 
            ints = PySCFIntegrals(mol)
            from quantel.wfn.csf import CSF as WFN 
        else: 
            ints = PySCFIntegrals(mol, 'scan')
            from quantel.wfn.roks import ROKS as WFN 
        
        # Initialise object for closed shell reference 
        mf = WFN(ints,'cs') 
        mf.get_orbital_guess(method="gwh") 
        LBFGS().run(mf, thresh=1e-7)
        # Localise orbitals in shellis 
        mf.localise() 
       
        # Initialise and optimise open shell singlet state  
        wfn = WFN(ints, '+-', mom_method=mom_method, scale_core_dens=True)
        # Select reference orbitals
        wfn.initialise(mo_guess=mf.mo_coeff) 
        #wfn.get_orbital_guess(method='rohf',reorder=True)
        #Cguess = wfn.mo_coeff.copy()
        #wfn.update() 
        DIIS().run(wfn, thresh=1e-7, maxit=500, max_vec=6)
        wfn.canonicalize()
        wfn.get_davidson_hessian_index(approx_hess=False)
        


