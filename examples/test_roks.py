import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals, PySCF_MO_Integrals
from quantel.wfn.roks import ROKS
from quantel.wfn.csf import CSF
from quantel.wfn.cisolver import CustomCI
from quantel.opt.gmf import GMF
from quantel.opt.lbfgs import LBFGS

molxyz = "allyl.xyz"
#molxyz = "h6.xyz"
#molxyz = "h3.xyz"

# Test RHF object with a range of optimisers
print("Test CSF object with a range of optimisers")
np.set_printoptions(linewidth=100000,precision=6,suppress=True)

for driver in ["pyscf"]:
    print("\n===============================================")
    print(f" Testing '{driver}' integral method")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule(molxyz, "cc-pvdz", "angstrom",spin=1)
    ints = PySCFIntegrals(mol,xc='tpss')
    hf_ints = PySCFIntegrals(mol)

    # Check against CSF
    wfn = ROKS(ints, '+')
    wfn.get_orbital_guess(method="rohf",reorder=False)
    LBFGS().run(wfn)
    # Perform excitation
    cguess = wfn.mo_coeff.copy()
    # HOMO/SOMO
    cguess[:,[wfn.ncore-1,wfn.ncore]] = cguess[:,[wfn.ncore,wfn.ncore-1]]
    # SOMO/LUMO
    #cguess[:,[wfn.ncore+1,wfn.ncore]] = cguess[:,[wfn.ncore,wfn.ncore+1]]
    wfn.initialise(cguess)
    # Analytic Hessian would probably be much faster for CSF/ROKS
    # Or at least analytic Hessian on vector
    GMF().run(wfn,index=1,maxit=10000)
    quit()
    cguess = wfn.mo_coeff.copy()
    # Order as HOMO/LUMO/SOMO
    cguess[:,[wfn.ncore-1,wfn.ncore,wfn.ncore+1]] = cguess[:,[wfn.ncore-1,wfn.ncore+1,wfn.ncore]]
    for i in range(10):
        excited1 = ROKS(ints, '++-')
        excited1.initialise(cguess)
        excited1.take_step(np.random.randn(excited1.nrot))
        LBFGS().run(excited1)
    for i in range(10):
        excited2 = ROKS(ints, '+-+')
        excited2.initialise(cguess)
        excited2.take_step(np.random.randn(excited2.nrot))
        LBFGS().run(excited2)
    #print(wfn.gradient)
    #print(wfn.get_numerical_gradient())
    #hess = np.zeros((wfn.nmo,wfn.nmo))
    #hess[wfn.rot_idx] = np.diag(wfn.get_numerical_hessian(diag=True))
    #print("Preconditioner:")
    #prec = np.zeros((wfn.nmo,wfn.nmo))
    #prec[wfn.rot_idx] = wfn.get_preconditioner()
    #print(prec)
    #print("Numerical Hessian:")
    #print(hess)
    



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


