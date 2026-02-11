from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.opt.hybrid_ef import HybridEF
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

    # Initialise RHF wavefunction with GWH orbital guess
    wfn = RHF(ints)
    wfn.get_orbital_guess('gwh')

    # Here, as an example, we climb a ladder of increasing Hessian index.
    # This demonstrates how eigenvector following can move away from a solution
    # with the wrong target Hessian index.
    for index in range(4):
        opt.run(wfn,index=index,approx_hess=False,plev=1)
        wfn.get_davidson_hessian_index(approx_hess=False)
