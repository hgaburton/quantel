from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.opt.hybrid_ef import HybridEF
from quantel.wfn.uhf import UHF
from quantel.wfn.rhf import RHF

if __name__ == "__main__":
    print("===============================================")
    print(f" Testing HybridEF optimisation method")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule("mol/allyl.xyz", "aug-cc-pvdz", "angstrom",charge=1,spin=0)
    ints = PySCFIntegrals(mol,with_df=True)

    # Initialise the optimiser
    opt = HybridEF()

    # Test for UHF wavefunction
    wfn = RHF(ints)
    wfn.get_orbital_guess('gwh')
    opt.run(wfn,index=2,approx_hess=False,plev=1)

    # Check Hessian index
    wfn.get_davidson_hessian_index(approx_hess=False)