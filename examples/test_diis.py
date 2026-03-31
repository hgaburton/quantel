from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.opt.lbfgs import LBFGS
from quantel.opt.diis import DIIS
from quantel.utils.linalg import matrix_print
import numpy as np
np.set_printoptions(linewidth=1000,precision=6,suppress=True)

# Set Occupation selector 
mom_methods = ["Aufbau", "MOM", "IMOM"] 

if __name__ == "__main__":
    print("\n===============================================")
    print(f" Testing DIIS opt. on CSF and ROKS objects")
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
        mf.localise_orbitals() 
        for mom_method in mom_methods: 
            print("\n===============================================")
            print(f" Testing Method: {mom_method}")
            print("===============================================")
            wfn = WFN(ints, '+-', mom_method=mom_method, scale_core_dens=True)
            wfn.initialise(mo_guess=mf.mo_coeff) 
            DIIS(max_vec=6).run(wfn, thresh=1e-7, maxit=500)
            #wfn.canonicalize()
            #wfn.get_davidson_hessian_index(approx_hess=False)
        


