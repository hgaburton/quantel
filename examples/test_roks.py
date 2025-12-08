import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals, PySCF_MO_Integrals
from quantel.wfn.roks import ROKS
from quantel.wfn.csf import CSF
from quantel.wfn.cisolver import CustomCI
from quantel.opt.lbfgs import LBFGS

molxyz = "formaldehyde.xyz"
molxyz = "h6.xyz"
molxyz = "h3.xyz"

# Test RHF object with a range of optimisers
print("Test CSF object with a range of optimisers")
np.set_printoptions(linewidth=100000,precision=6,suppress=True)

for driver in ["pyscf"]:
    print("\n===============================================")
    print(f" Testing '{driver}' integral method")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule(molxyz, "6-31g", "angstrom",spin=1)
    ints = PySCFIntegrals(mol,xc='0.5*HF')
    hf_ints = PySCFIntegrals(mol)

    # Check against CSF
    wfn = CSF(hf_ints, '+')
    wfn.get_orbital_guess(method="core",reorder=False)
    wfn.print(verbose=1)

    wfn = ROKS(ints, '+')
    wfn.get_orbital_guess(method="core",reorder=False)
    wfn.print()
    print(wfn.gradient)
    print(wfn.get_numerical_gradient())
    hess = np.diag(wfn.get_numerical_hessian())
    print("Numerical Hessian:")
    print(hess)
    print("Preconditioner:")
    print(wfn.get_preconditioner())
    quit()
    
    wfn.print(verbose=1)
    LBFGS().run(wfn)


    quit()

    # Initialise CSF object for an open-shell singlet state
    wfn = ROKS(ints, '+-+-')
    wfn.get_orbital_guess(method="core",reorder=False)
    np.random.seed(1)
    wfn.take_step(np.random.randn(wfn.nrot))

    wfn.print(verbose=1000)
    print(wfn.energy)
    print(wfn.energy_components)


    wfn.print()
    hs_coeff = wfn.mo_coeff.copy()

    for sc in ['++++++','+-+-+-','+++---','++-+--']:
        print("Spin coupling:", sc)
        wfn = ROKS(ints, sc)
        wfn.initialise(hs_coeff)
        wfn.take_step(np.random.randn(wfn.nrot))
        LBFGS().run(wfn)
        wfn.print()
    quit()

    # Setup optimiser
    for guess in ("gwh", "core"):
        print("\n************************************************")
        print(f" Testing '{guess}' initial guess method")
        print("************************************************")
        from quantel.opt.lbfgs import LBFGS
        wfn.get_orbital_guess(method="gwh")
        LBFGS().run(wfn)
        
        # Test canonicalisation 
        wfn.canonicalize()
        # Test Hessian index
        wfn.get_davidson_hessian_index()

    mo_ints = PySCF_MO_Integrals(ints) 
    mo_ints.update_orbitals(wfn.mo_coeff, wfn.ncore, wfn.nopen)
    Vc = mo_ints.scalar_potential()
    oei = mo_ints.oei_matrix(True)
    tei = mo_ints.tei_array(True,False)
    nmo = mo_ints.nact()

    ci_ints = quantel.MOintegrals(Vc,oei,tei,nmo)
    det_list = quantel.utils.csf_utils.csf_det_list(wfn.spin_coupling)
    ci = CustomCI(ci_ints, det_list, (1,1))
    e,x = ci.solve(10)
    print(np.sum(e))


