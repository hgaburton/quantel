from quantel.ints.pyscf_integrals import PySCF_MO_Integrals, PySCFIntegrals, PySCFMolecule
from quantel.wfn.rhf import RHF
from quantel.wfn.gUGA_cisolver import FCI
from quantel.opt.lbfgs import LBFGS

## Benchmark PySCF calculation 
from pyscf import gto, scf, fci
mol = gto.Mole()
geom="./mol/h6.xyz"
basis="sto-3g"
unit="angstrom" 

###


# Setup FCI
totspin = 0 
if __name__ == "__main__":
    # Setup molecule
    mol = PySCFMolecule(geom,basis,unit,spin=0,charge=0)
    ints = PySCFIntegrals(mol)

    # Run RHF to get MO coefficients
    wfn = RHF(ints)
    wfn.get_orbital_guess(method="gwh")
    LBFGS().run(wfn)

    # Build the integrals
    mo_ints = PySCF_MO_Integrals(ints)
    mo_ints.update_orbitals(wfn.mo_coeff,0,ints.nmo())

    # Setup and solve FCI
    ci = FCI(mo_ints, sum(mol.nelec), totspin)
    x, eci = ci.solve(4,verbose=5,maxit=1000)
    

    # PySCF Benchmark Calculation 
    with open(geom, 'r') as f:
        lines = f.readlines()
    mol.atom = ''.join(lines[2:])  
    mol.basis = basis 
    mol.unit = unit
    mol.build()
    mol.verbose = 0 
    mf = scf.RHF(mol).run()
    cisolver = fci.FCI(mf)
    cisolver.spin = 2*totspin 
    cisolver.nroots = 4 
    pyscf_e, civec = cisolver.kernel()
    print("  ----------------------------") 
    print("  Final gUGA FCI Results") 
    for i in range(eci.shape[1]): 
        print(f"  State {i}: {x[i]}, S2: {ci.get_s2(eci[:,i].copy())}")      
    
    # Verify solution
    import numpy
    if numpy.all(abs(pyscf_e - x) < 1e-6):
        raise ValueError("FCI energy does not match reference value")
    else:
        print("  ----------------------------") 
        print("--> Matches PySCF reference value")  
