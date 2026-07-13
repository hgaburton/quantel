from quantel.ints.pyscf_integrals import PySCF_MO_Integrals, PySCFIntegrals, PySCFMolecule
from quantel.wfn.rhf import RHF
from quantel.wfn.cisolver import FCI
from quantel.opt.diis import DIIS
from quantel.opt.lbfgs import LBFGS
from pyscf import gto, scf, fci

mol = gto.Mole()
with open('/home/camhuangbrown/code/quantel/examples/mol/h6.xyz', 'r') as f:
    lines = f.readlines()
mol.atom = ''.join(lines[2:])  # skip first two lines
mol.basis = 'sto-3g'
mol.unit = 'angstrom'
mol.build()

mf = scf.RHF(mol).run()

cisolver = fci.FCI(mf)
cisolver.nroots = 3 
e, civec = cisolver.kernel()
print(f"FCI energy: {e[0]:.10f} Hartree")

if __name__ == "__main__":
    # Setup molecule
    mol = PySCFMolecule("/home/camhuangbrown/code/quantel/examples/mol/h6.xyz", "sto-3g", "angstrom",spin=0,charge=0)
    ints = PySCFIntegrals(mol)

    # Run RHF to get MO coefficients
    wfn = RHF(ints)
    wfn.get_orbital_guess(method="gwh")
    #DIIS().run(wfn)
    LBFGS().run(wfn)

    # Build the integrals
    #Ccore = wfn.mo_coeff[:,0:0]
    #Cact = wfn.mo_coeff[:,0:ints.nmo()]
    mo_ints = PySCF_MO_Integrals(ints)
    mo_ints.update_orbitals(wfn.mo_coeff,0,ints.nmo())

    # Setup and solve FCI
    ci = FCI(mo_ints, (mol.nalfa(), mol.nbeta()), version=1)
    x, eci = ci.solve(3,verbose=5,maxit=1000)
    
    for i in range(eci.shape[1]): 
        print(f"Energy: {x[i]}, S2: {ci.get_s2(eci[:,i].copy())}")      

    print("PySCF results: ", e) 
    # Verify solution
    if abs(e[0] - x[0]) > 1e-6:
        raise ValueError("FCI energy does not match reference value")
    else:
        print("Matches PySCF reference value")  
